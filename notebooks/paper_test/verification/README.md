# Paper result verification

Run verify_all_paper_claims.ipynb to inspect all committed artifacts, or verify_table_01 through verify_table_09 for one table. verify_paper.py holds the shared checks. A match validates the named artifact and arithmetic only; it does not establish training provenance.

First run ../00_colab_runtime_check.ipynb. On a Mac, start in a clone containing paper/, Semester/, and notebooks/. On Colab, upload the clone to /content/CLIP-Animal-Classifier. The experiment notebooks ../cupl_official_prompts_oxford.ipynb and ../stanford_dualstream_recheck.ipynb select MPS when available.
