# Credit Default ML Project

## Setup

1. Clone the repo and go into the folder:
   
2. Create and activate a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

3. Install the libraries:
   ```
   pip install -r requirements.txt
   ```

4. Put the dataset at `data/train.csv` (re-name to project dataset to train.csv)

5. Turn on notebook output stripping (keeps git diffs clean):
   ```
   nbstripout --install
   ```

6. Use Jupyter or some other viewer and open a notebook in `notebooks/`:
   ```
   jupyter lab
   ```