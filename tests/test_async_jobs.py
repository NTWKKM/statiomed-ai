"""
tests/test_async_jobs.py - Unit Tests for HF Async Jobs & Storage Bucket Persistence
"""

from agent.async_jobs import HFJobRunner


def test_hf_job_runner_submit_and_status():
    runner = HFJobRunner(bucket_name="stat-shiny-storage")
    assert runner.bucket_name == "stat-shiny-storage"

    params = {"m_imputations": 50, "dataset_size": 15000, "target_var": "mortality_1yr"}
    job_info = runner.submit_job(job_type="heavy_mice", params=params)

    assert "job_heavy_mice_" in job_info["job_id"]
    assert job_info["status"] in ["COMPLETED", "RUNNING", "QUEUED"]
    assert "stat-shiny-storage" in job_info["bucket_mount"]

    status = runner.get_job_status(job_info["job_id"])
    assert status["job_id"] == job_info["job_id"]
    assert status["status"] in ["COMPLETED", "RUNNING", "QUEUED"]


def test_hf_job_runner_not_found():
    runner = HFJobRunner()
    status = runner.get_job_status("non_existent_job_123")
    assert status["status"] == "NOT_FOUND"
