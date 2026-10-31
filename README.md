# Loan Agent

Reads loan application forms (salary certificate images), checks them against payroll records,
and reports accuracy trends. The OCR uses DeepSeek-OCR when a GPU and model are available,
otherwise Tesseract.

## Files
- `app.py`: Streamlit interface (upload, dashboard, AI adjustment review).
- `loan_core.py`: OCR, field extraction, decision rules, logs, overrides.
- `monitor.py`: acceptance-rate monitor. Compares clear rates and makes one capped threshold step, or escalates.
- `tune_rules.py`: scores each fraud rule against `labels.csv`, writes `proposals.json` for human review.
- `check_form.ipynb`: notebook version of the form checker.
- `payroll.csv`: mock payroll and obligations records. Synthetic data only.
- `thresholds.json`: current thresholds. `high_dbr` is in shadow mode (not used for decisions yet).

## Run
1. Install: `python -m pip install -r requirements.txt`
2. Put the DeepSeek-OCR model files in this folder (optional).
3. Run the interface: `streamlit run app.py`
4. Run the monitor: `python monitor.py`

## Notes
- The loan type is chosen in the interface, not read from the form.
- `high_dbr` uses the system's obligations (`shadow_dbr`). Its limit is not verified against SAMA by income bracket yet.
- Sample form images are not included. Create your own fictional test forms.
