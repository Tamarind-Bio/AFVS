# Failed Subjob Recovery Guide

## Overview

When running AFVS on AWS Batch with spot instances, subjobs may fail after exhausting all retry attempts. These failures result in lost ligand screening data. This guide explains how to identify and recover failed subjobs using the recovery system.

## Problem

- **Current behavior**: Failed subjobs are tracked but NOT included in results
- **Impact**: Ligands in failed subjobs are silently excluded from final screening results
- **Root cause**: Spot interruptions after 4 retry attempts, or other persistent failures

## Solution: Recovery Workflow

### Step 1: Set Up On-Demand Queue (One-Time Setup)

The on-demand queue uses EC2 on-demand instances (not spot) for more reliable execution of previously failed subjobs.

#### 1.1 Get Resource IDs from Main VF Stack

```bash
cd cfn/

# Run helper script to get resource IDs
./helper-get-vf-resources.sh us-east-1

# Copy the output JSON and save it to:
# params/us-east-1/vf-ondemand-parameters.json
```

#### 1.2 Deploy the On-Demand Queue

```bash
# Create the on-demand queue stack
./03a-create-ondemand-queue.sh us-east-1

# Check deployment status (wait for CREATE_COMPLETE)
./03b-create-ondemand-queue-status.sh us-east-1
```

### Step 2: Identify Failed Subjobs

```bash
cd tools/

# Report only (no resubmission)
./afvs_recover_failed_subjobs.py --report-only
```

**Example output:**
```
Analyzing status file for failed subjobs...

Found 15 failed subjobs:
Workunit ID     Subjob ID    Expected Ligands     Collections
--------------------------------------------------------------------------------
5               3            50000                AADAA_0000001, AADAA_0000002
5               7            50000                AADAB_0000001
12              0            48500                AADAC_0000003, AADAC_0000004
...

Total expected ligands in failed subjobs: 750,000
```

### Step 3: Recover Failed Subjobs

#### Option A: Submit to On-Demand Queue (Recommended)

More reliable, uses EC2 on-demand instances:

```bash
./afvs_recover_failed_subjobs.py --queue vf-queue-ondemand
```

#### Option B: Submit to Default Spot Queue

Lower cost, but may fail again due to spot interruptions:

```bash
./afvs_recover_failed_subjobs.py
```

**Output:**
```
Creating recovery workunits (max 100 subjobs per workunit)...
  Creating recovery workunit A-1 with 15 subjobs...
  Submitting recovery workunit A-1 to AWS Batch...
  ✓ Submitted as job abc-123-def to queue vf-queue-ondemand

✓ Recovery status saved to ../workflow/status.recovery.json
✓ Successfully created and submitted 1 recovery workunit(s)
  Total failed subjobs being recovered: 15
  Queue used: vf-queue-ondemand
```

### Step 4: Monitor Recovery Progress

```bash
# Check recovery job status
cd tools/
./afvs_get_status.py

# Check recovery-specific status
cat ../workflow/status.recovery.json
```

### Step 5: Verify Recovery Results

After recovery jobs complete:

1. Results will be in the standard output locations:
   - Summary: `s3://{bucket}/{prefix}/summary/recovery-{id}/{subjob_id}.json.gz`
   - Parquet: `s3://{bucket}/{prefix}/{scenario}/parquet/recovery-{id}/{subjob_id}.parquet`

2. You can combine recovery results with main results using standard data processing tools

## Script Options

### `afvs_recover_failed_subjobs.py`

```bash
# Full usage
./afvs_recover_failed_subjobs.py \
  --status-file ../workflow/status.json \
  --config-file ../workflow/config.json \
  --queue vf-queue-ondemand \
  --max-subjobs-per-workunit 100 \
  --report-only
```

**Options:**
- `--status-file`: Path to status.json (default: `../workflow/status.json`)
- `--config-file`: Path to config.json (default: `../workflow/config.json`)
- `--report-only`: Only report failed subjobs, don't resubmit
- `--queue`: AWS Batch queue name (default: uses spot queue1)
- `--max-subjobs-per-workunit`: Max subjobs per recovery workunit (default: 100)

## Architecture Details

### How Recovery Works

1. **Identification**: Script scans `status.json` for subjobs with `status: FAILED`

2. **Workunit Creation**:
   - Failed subjobs are grouped into new "recovery workunits"
   - Each recovery workunit contains up to 100 failed subjobs
   - Original collection data is preserved

3. **Submission**:
   - Recovery workunits submitted as AWS Batch array jobs
   - Can target specific queue (on-demand vs. spot)
   - Uses same job definition as main workflow

4. **Tracking**:
   - Recovery metadata saved to `status.recovery.json`
   - Includes mapping from recovery subjob → original workunit/subjob

### Recovery Workunit Structure

```json
{
  "recovery_workunits": {
    "A-1": {
      "job_id": "abc-123-def",
      "queue": "vf-queue-ondemand",
      "subjobs_count": 15,
      "subjobs": {
        "0": {"ligands_expected": 50000},
        "1": {"ligands_expected": 50000},
        ...
      },
      "recovery_subjobs_mapping": {
        "0": {
          "original_workunit_id": "5",
          "original_subjob_id": "3"
        },
        ...
      }
    }
  }
}
```

## Cost Considerations

### Spot Queue (Default)
- **Pro**: Lower cost (~70% savings)
- **Con**: May fail again due to interruptions
- **Use when**: Failures were due to transient issues, not spot interruptions

### On-Demand Queue
- **Pro**: Reliable, no spot interruptions
- **Con**: Higher cost (full on-demand pricing)
- **Use when**: Failures were due to repeated spot interruptions

## Troubleshooting

### Recovery Jobs Also Failing
1. Check CloudWatch logs for the recovery job
2. Verify the collections in S3 are accessible
3. Consider increasing timeout in job definition
4. Check if specific collections are corrupted

### On-Demand Queue Not Available
Run the helper script to verify resources:
```bash
./helper-get-vf-resources.sh us-east-1
```

Then redeploy the on-demand queue stack if needed.

## Files Created

- `/tools/afvs_recover_failed_subjobs.py` - Main recovery script
- `/cfn/yaml/vf-ondemand-queue.yaml` - CloudFormation template for on-demand queue
- `/cfn/03a-create-ondemand-queue.sh` - Deployment script
- `/cfn/03b-create-ondemand-queue-status.sh` - Status check script
- `/cfn/helper-get-vf-resources.sh` - Helper to get resource IDs
- `/cfn/params/{region}/vf-ondemand-parameters.json` - Parameter files

## Summary

This recovery system allows you to:
1. Identify all failed subjobs and their expected ligand counts
2. Resubmit failed subjobs as new recovery workunits
3. Target reliable on-demand instances to avoid repeat failures
4. Track recovery progress separately from main workflow
5. Preserve complete screening coverage despite spot interruptions
