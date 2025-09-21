# How to Use and Extend main.py

## Running the Script
1. Make sure your dataset (e.g., `conn.log.labeled`) is in the project directory.
2. Run the script with:
	```bash
	python3 main.py
	```
3. The script will preprocess the data, train models, and print results/plots.

## Extending with New Datasets
- Ensure new datasets have similar columns or adjust the `selected_cols` and `rename_map` in `main.py` as needed.
- Follow the same preprocessing steps for consistency (missing value handling, encoding, etc.).
- You can combine multiple datasets by loading and concatenating them with pandas before preprocessing.
- Document any changes you make for clarity and reproducibility.

## Collaboration Tips
- Keep code modular and well-commented.
- Use version control (git) to track changes.
- Communicate major changes to the team and update this README as needed.

If you have questions or improvements, please document them and share with the team!
