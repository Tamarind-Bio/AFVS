"""afvs_get_status.py in job_storage_mode=presigned.

The docking jobs carry no AWS identity and write only through presigned URLs,
so nothing they wrote is trusted for status or billing: states and vCPU time
come from AWS Batch DescribeJobs alone, and no S3 client is ever created.
Reuses the boto3 recorder and the S3 screen harness of test_awsbatch_unchanged."""

import io
import json
import subprocess
import sys
import time
from contextlib import redirect_stdout

import pytest

from test_awsbatch_unchanged import boto, batch_tree, load, prepare_batch  # noqa: F401 (boto is autouse)


# The lines of the Tamarind wrapper's parser that read this report
WRAPPER_PARSE = r'''
output=$(cat)
submitted=$(echo "$output" | awk '/SUBMITTED/ {print $2}')
runnable=$(echo "$output" | awk '/RUNNABLE/ {print $2}')
pending=$(echo "$output" | awk '/PENDING/ {print $2}')
running=$(echo "$output" | awk '/RUNNING/ {print $2}')
starting=$(echo "$output" | awk '/STARTING/ {print $2}')
succeeded=$(echo "$output" | awk '/SUCCEEDED/ {print $2}')
failed=$(echo "$output" | awk '/FAILED/ {print $2}')
total_succeeded=$(echo "$output" | awk '/- Succeeded/ {print $NF}')
total_failed=$(echo "$output" | awk '/- Failed/ {print $NF}')
total_skipped=$(echo "$output" | awk '/Skipped ligands/ {print $NF}')
vcpu_per_docking=$(echo "$output" | awk '/vCPU seconds per docking/ {print $NF}')
vcpu_total=$(echo "$output" | awk '/vCPU hours total/ {print $NF}')
vcpu_interrupted=$(echo "$output" | awk '/vCPU hours interrupted/ {print $NF}')
complete=0
if [[ "$submitted" == "0" ]] && [[ "$runnable" == "0" ]] && [[ "$pending" == "0" ]] && \
   [[ "$running" == "0" ]] && [[ "$starting" == "0" ]]; then
    complete=1
fi
for v in submitted runnable pending running starting succeeded failed total_succeeded \
         total_failed total_skipped vcpu_per_docking vcpu_total vcpu_interrupted complete; do
    printf '%s=%s\n' "$v" "${!v}"
done
'''


def wrapper_parse(status_output):
    ret = subprocess.run(["bash", "-c", WRAPPER_PARSE], input=status_output,
                         capture_output=True, text=True, check=True)
    return dict(line.split("=", 1) for line in ret.stdout.splitlines())


STATUSES = ['SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING', 'SUCCEEDED', 'FAILED']


def expected_report(table, active_vcpus, vcpu_hours, vcpu_hours_interrupted):
    rows = "".join(f"{state:>15}   {table.get(state, (0, 0, 0))[0]:^10}   {table.get(state, (0, 0, 0))[1]:^15}   "
                   f"{table.get(state, (0, 0, 0))[2]:^15}\n" for state in STATUSES)
    return (
        "Getting workunit status\n"
        "Getting subjob status\n"
        "-----------------------------------------------------------------\n"
        "AWS Batch Progress\n"
        "-----------------------------------------------------------------\n"
        "\n"
        "         Status   Workunits        Subjobs      Dockings (est.)\n"
        f"{rows}"
        "\n"
        f"Active vCPUs: {active_vcpus}\n"
        "\n"
        "-----------------------------------------------------------------\n"
        "Completed Summary\n"
        "-----------------------------------------------------------------\n"
        "\n"
        "Docking counts are not read here; they come from the tool container's progress.json\n"
        "\n"
        "* vCPU seconds per docking: N/A\n"
        f"* vCPU hours total        : {vcpu_hours}\n"
        f"* vCPU hours interrupted  : {vcpu_hours_interrupted}\n"
        "\n"
    )


@pytest.fixture
def screen(tmp_path, boto, monkeypatch):
    """A presigned screen of three single-subjob workunits, submitted as j1..j3."""
    tools = batch_tree(tmp_path)
    ctrl = tools / "templates" / "all.ctrl"
    ctrl.write_text(ctrl.read_text().replace("job_storage_mode=s3\n", "job_storage_mode=presigned\n")
                    + "presigned_url_expiry_seconds=43200\n")
    prepare_batch(tools, boto, monkeypatch)

    workflow = tools.parent / "workflow"
    config = json.loads((workflow / "config.json").read_text())
    assert (config['job_storage_mode'], config['batchsystem']) == ("presigned", "awsbatch")

    status = json.loads((workflow / "status.json").read_text())
    assert list(status['workunits']) == ["1", "2", "3"]
    for index, workunit in enumerate(status['workunits'].values(), start=1):
        workunit['status'] = "SUBMITTED"
        workunit['job_id'] = f"j{index}"
        workunit['subjobs']['0']['status'] = "SUBMITTED"
    (workflow / "status.json").write_text(json.dumps(status))

    module = load("afvs_get_status.py", monkeypatch)

    def run(*args):
        boto.calls.clear()
        monkeypatch.setattr(sys, "argv", ["afvs_get_status.py", *args])
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                module.main()
        finally:
            run.output = out.getvalue()
        return run.output

    run.workflow = workflow
    return run


def parent(job_id, state):
    return {'jobId': job_id, 'status': state, 'arrayProperties': {'statusSummary': {state: 1}}}


def vcpu_requirement(vcpus):
    return {'resourceRequirements': [{'type': 'MEMORY', 'value': '32000'}, {'type': 'VCPU', 'value': str(vcpus)}]}


FIRST_ROUND = [
    {'jobs': [parent("j1", "SUCCEEDED"), parent("j2", "FAILED"), parent("j3", "RUNNING")]},
    {'jobs': [
        # 60 s x 16 vCPUs (from resourceRequirements)
        {'jobId': "j1:0", 'status': "SUCCEEDED", 'container': vcpu_requirement(16),
         'attempts': [{'startedAt': 3000, 'stoppedAt': 63000}]},
        # two 30 s attempts x 8 vCPUs (container.vcpus fallback): all billed, the first interrupted
        {'jobId': "j2:0", 'status': "FAILED", 'container': {'vcpus': 8},
         'attempts': [{'startedAt': 5000, 'stoppedAt': 35000}, {'startedAt': 40000, 'stoppedAt': 70000}]},
        {'jobId': "j3:0", 'status': "RUNNING", 'container': vcpu_requirement(32),
         'attempts': []},
    ]},
]


def test_presigned_status_from_describe_jobs_only(screen, boto):
    boto.describe_jobs_responses = list(FIRST_ROUND)
    out = screen()

    # 960 + 480 vCPU seconds billed, 240 of them interrupted; j3 runs on 32 vCPUs
    assert out == expected_report({'RUNNING': (1, 1, 3), 'SUCCEEDED': (1, 1, 3), 'FAILED': (1, 1, 3)},
                                  32, "0.40", "0.07")
    # Only Batch: no S3 client, no download. The size-1 arrays were submitted
    # as size 2, so each padding child (index 1) is asked for too, to bill it
    assert boto.calls == [
        ('client', 'batch', {'region_name': 'us-west-2'}),
        ('describe_jobs', ["j1", "j2", "j3"]),
        ('describe_jobs', ["j1:0", "j1:1", "j2:0", "j2:1", "j3:0", "j3:1"]),
    ]

    after = json.loads((screen.workflow / "status.json").read_text())
    subjobs = [w['subjobs']['0'] for w in after['workunits'].values()]
    assert [(s['status'], s.get('vcpu_seconds'), s.get('vcpu_seconds_interrupted')) for s in subjobs] == [
        ("SUCCEEDED", 960.0, 0), ("FAILED", 480.0, 240.0), ("RUNNING", None, None)]
    assert (after['summary']['vcpu_seconds'], after['summary']['vcpu_seconds_interrupted']) == (1440.0, 240.0)
    assert not (screen.workflow / "status.tmp.json").exists()

    # The next call asks only about what is not final yet, and bills it once
    boto.describe_jobs_responses = [
        {'jobs': [parent("j3", "SUCCEEDED")]},
        {'jobs': [{'jobId': "j3:0", 'status': "SUCCEEDED", 'container': vcpu_requirement(32),
                   'attempts': [{'startedAt': 0, 'stoppedAt': 45000}]}]},
    ]
    out = screen()
    assert boto.calls == [
        ('client', 'batch', {'region_name': 'us-west-2'}),
        ('describe_jobs', ["j3"]),
        ('describe_jobs', ["j3:0", "j3:1"]),
    ]
    # + 45 s x 32 vCPUs
    assert out == expected_report({'SUCCEEDED': (2, 2, 6), 'FAILED': (1, 1, 3)}, 0, "0.80", "0.07")


def test_presigned_recovery_round_overrides_and_is_billed(screen, boto):
    boto.describe_jobs_responses = list(FIRST_ROUND)
    screen()

    (screen.workflow / "status.recovery.json").write_text(json.dumps({'recovery_workunits': {
        "a-r1-1": {
            'round': 1, 'job_id': "r1", 'queue': "vf-queue1", 'subjobs_count': 1,
            's3_download_path': "scores/someone/dataset/batch-job/input/recovery/a-r1-1.tar.gz",
            'subjobs': {"0": {'ligands_expected': 3}},
            'recovery_subjobs_mapping': {"0": {'original_workunit_id': "2", 'original_subjob_id': "0"}},
        },
    }}))

    boto.describe_jobs_responses = [
        {'jobs': [parent("j3", "RUNNING"), parent("r1", "RUNNING")]},
        {'jobs': [
            {'jobId': "j3:0", 'status': "RUNNING", 'container': vcpu_requirement(32), 'attempts': []},
            {'jobId': "r1:0", 'status': "STARTING", 'container': vcpu_requirement(16), 'attempts': []},
        ]},
    ]
    out = screen()
    assert boto.calls[1:] == [('describe_jobs', ["j3", "r1"]), ('describe_jobs', ["j3:0", "j3:1", "r1:0", "r1:1"])]
    # Workunit 2 follows its rerun subjob (STARTING), not its main FAILED
    assert out == expected_report({'STARTING': (1, 1, 3), 'RUNNING': (1, 1, 3), 'SUCCEEDED': (1, 1, 3)},
                                  48, "0.40", "0.07")

    boto.describe_jobs_responses = [
        {'jobs': [parent("j3", "SUCCEEDED"), parent("r1", "SUCCEEDED")]},
        {'jobs': [
            {'jobId': "j3:0", 'status': "SUCCEEDED", 'container': vcpu_requirement(32),
             'attempts': [{'startedAt': 0, 'stoppedAt': 45000}]},
            {'jobId': "r1:0", 'status': "SUCCEEDED", 'container': vcpu_requirement(16),
             'attempts': [{'startedAt': 0, 'stoppedAt': 30000}, {'startedAt': 31000, 'stoppedAt': 61000}]},
        ]},
    ]
    out = screen()
    # 1440 + 1440 (j3) + 960 (r1, both attempts) vCPU seconds; 240 + 480 interrupted
    expected = expected_report({'SUCCEEDED': (3, 3, 9)}, 0, "1.07", "0.20")
    assert out == expected

    main_status = json.loads((screen.workflow / "status.json").read_text())
    assert main_status['workunits']["2"]['subjobs']["0"]['status'] == "FAILED"
    assert main_status['overview_status']['SUCCEEDED'] == {'ligands': 9, 'workunits': 3, 'subjobs': 3}
    recovery = json.loads((screen.workflow / "status.recovery.json").read_text())
    rerun = recovery['recovery_workunits']["a-r1-1"]['subjobs']["0"]
    assert (rerun['status'], rerun['vcpu_seconds'], rerun['vcpu_seconds_interrupted']) == ("SUCCEEDED", 960.0, 480.0)
    assert not (screen.workflow / "status.recovery.tmp.json").exists()

    # --recovery shows the same view; nothing is left to ask Batch about
    assert screen("--recovery") == expected
    assert boto.calls == [('client', 'batch', {'region_name': 'us-west-2'})]

    parsed = wrapper_parse(expected)
    assert parsed == {
        'submitted': "0", 'runnable': "0", 'pending': "0", 'running': "0", 'starting': "0",
        'succeeded': "3", 'failed': "0",
        # docking counts are not in this report (they come from progress.json)
        'total_succeeded': "", 'total_failed': "", 'total_skipped': "",
        'vcpu_per_docking': "N/A", 'vcpu_total': "1.07", 'vcpu_interrupted': "0.20", 'complete': "1",
    }


def test_padding_child_is_billed_and_a_child_batch_never_made_fails(screen, boto):
    """The padding child of a size-1 array runs and is billed with its
    workunit. A child Batch never reports once its parent is final would
    otherwise sit SUBMITTED for good: it fails, unbilled."""
    boto.describe_jobs_responses = [
        {'jobs': [parent("j1", "SUCCEEDED"), parent("j2", "FAILED"), parent("j3", "SUCCEEDED")]},
        {'jobs': [
            {'jobId': "j1:0", 'status': "SUCCEEDED", 'container': vcpu_requirement(16),
             'attempts': [{'startedAt': 0, 'stoppedAt': 60000}]},
            # the padding child: 9 s x 16 vCPUs
            {'jobId': "j1:1", 'status': "SUCCEEDED", 'container': vcpu_requirement(16),
             'attempts': [{'startedAt': 0, 'stoppedAt': 9000}]},
            {'jobId': "j3:0", 'status': "SUCCEEDED", 'container': vcpu_requirement(16),
             'attempts': [{'startedAt': 0, 'stoppedAt': 60000}]},
            # j2's children never existed (its array failed before starting them)
        ]},
    ]
    out = screen()
    after = json.loads((screen.workflow / "status.json").read_text())
    assert after['workunits']["1"]['padding']['vcpu_seconds'] == 144.0
    assert after['workunits']["2"]['subjobs']["0"]['status'] == "FAILED"
    assert after['workunits']["2"]['subjobs']["0"]['vcpu_seconds'] == 0
    assert after['summary']['vcpu_seconds'] == 960.0 + 144.0 + 960.0
    assert wrapper_parse(out)['complete'] == "1" and wrapper_parse(out)['failed'] == "1"


def test_presigned_report_parses_as_in_progress(screen, boto):
    boto.describe_jobs_responses = list(FIRST_ROUND)
    assert wrapper_parse(screen()) == {
        'submitted': "0", 'runnable': "0", 'pending': "0", 'running': "1", 'starting': "0",
        'succeeded': "1", 'failed': "1",
        'total_succeeded': "", 'total_failed': "", 'total_skipped': "",
        'vcpu_per_docking': "N/A", 'vcpu_total': "0.40", 'vcpu_interrupted': "0.07", 'complete': "0",
    }


@pytest.mark.parametrize("job_storage_mode,batchsystem", [("presigned", "slurm"), ("sharedfs", "awsbatch")])
def test_unsupported_configuration_exits_nonzero(screen, boto, job_storage_mode, batchsystem):
    config_path = screen.workflow / "config.json"
    config = json.loads(config_path.read_text())
    config['job_storage_mode'] = job_storage_mode
    config['batchsystem'] = batchsystem
    config_path.write_text(json.dumps(config))

    with pytest.raises(SystemExit) as exited:
        screen()
    assert exited.value.code == 2
    assert screen.output == (f"Configuration with job_storage_mode={job_storage_mode} and "
                             f"batchsystem={batchsystem} is not supported\n")
    assert boto.calls == []


def recovery_entry(job_id, **extra):
    return {
        'round': 1, 'job_id': job_id, 'queue': "vf-queue1", 'subjobs_count': 1,
        's3_download_path': "scores/someone/dataset/batch-job/input/recovery/a-r1-1.tar.gz",
        'subjobs': {"0": {'ligands_expected': 3}},
        'recovery_subjobs_mapping': {"0": {'original_workunit_id': "2", 'original_subjob_id': "0"}},
        **extra,
    }


@pytest.mark.parametrize("recorded_ago, rerun_counts_as", [
    # Its job may exist and not be listed yet: in flight, as recovery counts it
    (60, "SUBMITTED"),
    # Too old to be in flight unseen: the previous attempt (FAILED) stands, so
    # recovery runs and abandons or adopts it
    (3600, "FAILED"),
])
def test_a_recovery_workunit_recorded_but_not_submitted_is_not_asked_about(screen, boto, recorded_ago, rerun_counts_as):
    """Recover records a workunit (job_id None) before SubmitJob. A status run
    in between, or after a crash there, must not ask Batch about job None,
    and must count its subjobs as recovery does."""
    boto.describe_jobs_responses = list(FIRST_ROUND)
    screen()
    (screen.workflow / "status.recovery.json").write_text(json.dumps({'recovery_workunits': {
        "a-r1-1": recovery_entry(None, recorded_at=time.time() - recorded_ago)}}))

    boto.describe_jobs_responses = [
        {'jobs': [parent("j3", "RUNNING")]},
        {'jobs': [{'jobId': "j3:0", 'status': "RUNNING", 'container': vcpu_requirement(32), 'attempts': []}]},
    ]
    screen()
    assert boto.calls[1:] == [('describe_jobs', ["j3"]), ('describe_jobs', ["j3:0", "j3:1"])]
    recovery = json.loads((screen.workflow / "status.recovery.json").read_text())
    assert recovery['recovery_workunits']["a-r1-1"]['job_id'] is None
    # Workunit 2's only subjob is the one rerun
    overview = json.loads((screen.workflow / "status.json").read_text())['overview_status']
    assert overview[rerun_counts_as]['subjobs'] == 1
    assert overview['SUCCEEDED']['subjobs'] == 1 and overview['RUNNING']['subjobs'] == 1


@pytest.mark.parametrize("submitted_ago, expected", [
    # DescribeJobs can lag a SubmitJob: a job just submitted is still coming
    (60, "SUBMITTED"),
    # Long gone (aged out of Batch): fail it closed, unbilled, so recovery reruns it
    (3600, "FAILED"),
])
def test_a_parent_missing_from_describe_jobs_fails_only_after_a_grace(screen, boto, submitted_ago, expected):
    status_file = screen.workflow / "status.json"
    status = json.loads(status_file.read_text())
    status['workunits']["3"]['submitted_at'] = time.time() - submitted_ago
    status_file.write_text(json.dumps(status))

    boto.describe_jobs_responses = [
        {'jobs': [parent("j1", "RUNNING"), parent("j2", "RUNNING")]},
        {'jobs': [{'jobId': f"j{n}:0", 'status': "RUNNING", 'container': vcpu_requirement(16), 'attempts': []}
                  for n in (1, 2)]},
    ]
    screen()
    subjob = json.loads(status_file.read_text())['workunits']["3"]['subjobs']["0"]
    assert subjob['status'] == expected
    assert subjob.get('vcpu_seconds') == (0 if expected == "FAILED" else None)


def test_status_reads_and_writes_under_the_lock(screen, boto, monkeypatch):
    """Status holds the lock from its read to its write-back, so an
    overlapping status run (or recover) that wrote a final state first is
    read, not overwritten with an older read."""
    boto.describe_jobs_responses = list(FIRST_ROUND)
    screen()
    recovery_file = screen.workflow / "status.recovery.json"
    recovery_file.write_text(json.dumps({'recovery_workunits': {"a-r1-1": recovery_entry("r1")}}))

    boto.describe_jobs_responses = [
        {'jobs': [parent("j3", "RUNNING"), parent("r1", "RUNNING")]},
        {'jobs': [
            {'jobId': "j3:0", 'status': "RUNNING", 'container': vcpu_requirement(32), 'attempts': []},
            {'jobId': "r1:0", 'status': "RUNNING", 'container': vcpu_requirement(16), 'attempts': []},
        ]},
    ]

    # Another run writes r1:0 FAILED (billed) while this one waits for the lock
    afvs_presign = sys.modules['afvs_presign']
    real_lock = afvs_presign.recovery_lock

    def lock_after_faster_run():
        done = recovery_entry("r1", status="FAILED")
        done['subjobs']["0"].update(status="FAILED", vcpus=16, vcpu_seconds=480.0, vcpu_seconds_interrupted=0)
        recovery_file.write_text(json.dumps({'recovery_workunits': {"a-r1-1": done}}))
        return real_lock()

    monkeypatch.setattr(afvs_presign, "recovery_lock", lock_after_faster_run)
    screen()
    # It read r1 as final, so asked Batch only about j3 and the padding
    # children still to bill (a stale read would ask about r1 and r1:0 too)
    assert boto.calls[1:] == [('describe_jobs', ["j3"]), ('describe_jobs', ["j3:0", "j3:1", "r1:1"])]
    entry = json.loads(recovery_file.read_text())['recovery_workunits']["a-r1-1"]
    assert entry['status'] == "FAILED"
    assert (entry['subjobs']["0"]['status'], entry['subjobs']["0"]['vcpu_seconds']) == ("FAILED", 480.0)
    # Its padding child never reported and its array is final: failed closed
    assert entry['padding']['status'] == "FAILED"


def test_fractional_vcpus_in_the_config_are_not_truncated_or_fatal(screen, boto):
    """Batch takes fractional vCPUs (Fargate): a running child is counted at
    the vCPUs Batch reports for it, or as configured when its container says
    nothing, never truncated and never fatal."""
    config_file = screen.workflow / "config.json"
    config = json.loads(config_file.read_text())
    config['aws_batch_subjob_vcpus'] = "0.5"
    config_file.write_text(json.dumps(config))

    boto.describe_jobs_responses = [
        {'jobs': [parent("j1", "RUNNING"), parent("j2", "RUNNING"), parent("j3", "RUNNING")]},
        {'jobs': [{'jobId': "j1:0", 'status': "RUNNING", 'container': vcpu_requirement(0.5), 'attempts': []},
                  {'jobId': "j2:0", 'status': "RUNNING", 'container': {}, 'attempts': []}]},
    ]
    out = screen()
    assert "Active vCPUs: 1.0" in out.splitlines()
    after = json.loads((screen.workflow / "status.json").read_text())
    assert [after['workunits'][n]['subjobs']['0'].get('vcpus') for n in ("1", "2")] == [0.5, 0.5]


def test_a_running_padding_child_counts_as_active(screen, boto):
    """A size-1 array runs two children; while the padding one runs, its
    vCPUs are in use too."""
    boto.describe_jobs_responses = [
        {'jobs': [parent("j1", "RUNNING"), parent("j2", "SUCCEEDED"), parent("j3", "SUCCEEDED")]},
        {'jobs': [
            {'jobId': "j1:0", 'status': "RUNNING", 'container': vcpu_requirement(16), 'attempts': []},
            {'jobId': "j1:1", 'status': "RUNNING", 'container': vcpu_requirement(16), 'attempts': []},
            {'jobId': "j2:0", 'status': "SUCCEEDED", 'container': vcpu_requirement(16),
             'attempts': [{'startedAt': 0, 'stoppedAt': 1000}]},
            {'jobId': "j2:1", 'status': "SUCCEEDED", 'container': vcpu_requirement(16),
             'attempts': [{'startedAt': 0, 'stoppedAt': 1000}]},
            {'jobId': "j3:0", 'status': "SUCCEEDED", 'container': vcpu_requirement(16),
             'attempts': [{'startedAt': 0, 'stoppedAt': 1000}]},
            {'jobId': "j3:1", 'status': "SUCCEEDED", 'container': vcpu_requirement(16),
             'attempts': [{'startedAt': 0, 'stoppedAt': 1000}]},
        ]},
    ]
    assert "Active vCPUs: 32" in screen().splitlines()
