-- Audit log and review queue (ER diagram, Figure 3.6 of the report)
CREATE TABLE IF NOT EXISTS dataset (
  dataset_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, num_samples INTEGER, manipulation_type TEXT);
CREATE TABLE IF NOT EXISTS app_user (
  user_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, role TEXT CHECK(role IN ('analyst','admin','auditor')), email TEXT);
CREATE TABLE IF NOT EXISTS model_version (
  model_id INTEGER PRIMARY KEY AUTOINCREMENT, arch_name TEXT, trained_on TEXT, metrics_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS media_sample (
  sample_id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id INTEGER REFERENCES dataset(dataset_id),
  file_path TEXT, label TEXT, duration_sec REAL);
CREATE TABLE IF NOT EXISTS inference_result (
  result_id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id INTEGER REFERENCES media_sample(sample_id),
  model_id INTEGER REFERENCES model_version(model_id), verdict TEXT, confidence_score REAL,
  uncertainty_score REAL, input_hash TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS explanation (
  explain_id INTEGER PRIMARY KEY AUTOINCREMENT, result_id INTEGER REFERENCES inference_result(result_id),
  gradcam_path TEXT, shap_path TEXT, modality TEXT);
CREATE TABLE IF NOT EXISTS review_queue (
  review_id INTEGER PRIMARY KEY AUTOINCREMENT, result_id INTEGER UNIQUE REFERENCES inference_result(result_id),
  status TEXT CHECK(status IN ('pending','approved','rejected','escalated')), reviewer_id INTEGER REFERENCES app_user(user_id),
  reviewed_at TEXT);
CREATE TABLE IF NOT EXISTS review_history (
  history_id INTEGER PRIMARY KEY AUTOINCREMENT,
  review_id INTEGER REFERENCES review_queue(review_id),
  status TEXT NOT NULL,
  reviewer TEXT,
  note TEXT,
  created_at TEXT NOT NULL
);
