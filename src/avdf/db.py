"""SQLite audit log and review queue (schema in db/schema.sql)."""
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = Path(__file__).resolve().parents[2] / "db" / "schema.sql"


class AuditDB:
    def __init__(self, path="db/audit.sqlite"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA.read_text())

    def _now(self):
        return datetime.now().isoformat(timespec="seconds")

    def register_model(self, arch, trained_on, metrics_json="{}"):
        """Return the id of this model version, inserting it only the first time it is seen."""
        row = self.con.execute("SELECT model_id FROM model_version WHERE arch_name=? AND trained_on=? ORDER BY model_id LIMIT 1",
                               (arch, trained_on)).fetchone()
        if row:
            return row[0]
        cur = self.con.execute("INSERT INTO model_version(arch_name, trained_on, metrics_json, created_at) VALUES (?,?,?,?)",
                               (arch, trained_on, metrics_json, self._now()))
        self.con.commit(); return cur.lastrowid

    def log_result(self, file_path, input_hash, verdict, confidence, uncertainty, model_id, duration=None):
        cur = self.con.execute("INSERT INTO media_sample(dataset_id, file_path, label, duration_sec) VALUES (NULL,?,NULL,?)", (file_path, duration))
        sid = cur.lastrowid
        cur = self.con.execute("INSERT INTO inference_result(sample_id, model_id, verdict, confidence_score, uncertainty_score, input_hash, created_at) "
                               "VALUES (?,?,?,?,?,?,?)", (sid, model_id, verdict, confidence, uncertainty, input_hash, self._now()))
        self.con.commit(); return cur.lastrowid

    def add_explanation(self, result_id, gradcam_path, shap_path):
        self.con.execute("INSERT INTO explanation(result_id, gradcam_path, shap_path, modality) VALUES (?,?,?,?)",
                         (result_id, gradcam_path, shap_path, "audio-visual"))
        self.con.commit()

    def enqueue(self, result_id):
        self.con.execute("INSERT INTO review_queue(result_id, status) VALUES (?, 'pending')", (result_id,))
        self.con.commit()

    def pending(self):
        return [dict(r) for r in self.con.execute(
            "SELECT q.review_id, r.result_id, r.verdict, r.confidence_score, r.uncertainty_score, s.file_path, r.created_at "
            "FROM review_queue q JOIN inference_result r USING(result_id) JOIN media_sample s USING(sample_id) "
            "WHERE q.status='pending' ORDER BY r.created_at DESC")]

    def resolve(self, review_id, status, reviewer_id=None, reviewer=None, note=""):
        assert status in ("approved", "rejected", "escalated")
        self.con.execute("UPDATE review_queue SET status=?, reviewer_id=?, reviewed_at=? WHERE review_id=?",
                         (status, reviewer_id, self._now(), review_id))
        self.con.execute("INSERT INTO review_history(review_id, status, reviewer, note, created_at) VALUES (?,?,?,?,?)",
                         (review_id, status, reviewer, note, self._now()))
        self.con.commit()

    def review_history(self, review_id):
        return [dict(r) for r in self.con.execute(
            "SELECT status, reviewer, note, created_at FROM review_history WHERE review_id=? ORDER BY created_at",
            (review_id,))]

    def _audit_where(self, query="", verdict="", review=""):
        where, args = ["(s.file_path LIKE ? OR r.input_hash LIKE ?)"], [f"%{query}%", f"%{query}%"]
        if verdict:
            where.append("r.verdict = ?"); args.append(verdict)
        if review == "auto":
            where.append("q.status IS NULL")
        elif review:
            where.append("q.status = ?"); args.append(review)
        return " AND ".join(where), args

    def audit(self, limit=100, offset=0, query="", verdict="", review=""):
        where, args = self._audit_where(query, verdict, review)
        return [dict(r) for r in self.con.execute(
            "SELECT r.result_id, r.verdict, r.confidence_score, r.uncertainty_score, r.input_hash, r.created_at, "
            "s.file_path, s.duration_sec, m.arch_name AS model_version, q.status AS review_status "
            "FROM inference_result r JOIN media_sample s USING(sample_id) JOIN model_version m USING(model_id) "
            f"LEFT JOIN review_queue q USING(result_id) WHERE {where} "
            "ORDER BY r.created_at DESC, r.result_id DESC LIMIT ? OFFSET ?", (*args, limit, offset))]

    def audit_count(self, query="", verdict="", review=""):
        where, args = self._audit_where(query, verdict, review)
        return self.con.execute(
            "SELECT COUNT(*) FROM inference_result r JOIN media_sample s USING(sample_id) "
            f"LEFT JOIN review_queue q USING(result_id) WHERE {where}", args).fetchone()[0]

    def case(self, result_id):
        """Everything known about one inference: result, media, model, explanation and review trail."""
        row = self.con.execute(
            "SELECT r.*, s.file_path, s.duration_sec, m.arch_name AS model_version, m.trained_on, "
            "e.gradcam_path, e.shap_path, q.review_id, q.status AS review_status, q.reviewed_at "
            "FROM inference_result r JOIN media_sample s USING(sample_id) JOIN model_version m USING(model_id) "
            "LEFT JOIN explanation e USING(result_id) LEFT JOIN review_queue q USING(result_id) "
            "WHERE r.result_id = ?", (result_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["review_history"] = self.review_history(out["review_id"]) if out["review_id"] else []
        return out

    def reviewed(self, limit=50):
        return [dict(r) for r in self.con.execute(
            "SELECT q.review_id, q.status, q.reviewed_at, r.result_id, r.verdict, r.confidence_score, r.uncertainty_score, "
            "s.file_path, h.reviewer, h.note FROM review_queue q JOIN inference_result r USING(result_id) "
            "JOIN media_sample s USING(sample_id) LEFT JOIN review_history h ON h.history_id = "
            "(SELECT MAX(history_id) FROM review_history WHERE review_id = q.review_id) "
            "WHERE q.status != 'pending' ORDER BY q.reviewed_at DESC LIMIT ?", (limit,))]

    def stats(self):
        """Aggregates for the overview dashboard."""
        one = lambda sql, *a: self.con.execute(sql, a).fetchone()[0]
        verdicts = dict(self.con.execute("SELECT verdict, COUNT(*) FROM inference_result GROUP BY verdict").fetchall())
        reviews = dict(self.con.execute("SELECT status, COUNT(*) FROM review_queue GROUP BY status").fetchall())
        total = one("SELECT COUNT(*) FROM inference_result")
        daily = [dict(day=d, n=n) for d, n in self.con.execute(
            "SELECT substr(created_at, 1, 10) AS day, COUNT(*) FROM inference_result GROUP BY day ORDER BY day DESC LIMIT 14")]
        points = [dict(r) for r in self.con.execute(
            "SELECT r.result_id, r.verdict, r.confidence_score, r.uncertainty_score, q.status AS review_status "
            "FROM inference_result r LEFT JOIN review_queue q USING(result_id) ORDER BY r.result_id DESC LIMIT 500")]
        return {"total": total, "verdicts": verdicts, "reviews": reviews,
                "routed": sum(reviews.values()), "auto_accepted": total - sum(reviews.values()),
                "mean_confidence": one("SELECT AVG(confidence_score) FROM inference_result"),
                "mean_uncertainty": one("SELECT AVG(uncertainty_score) FROM inference_result"),
                "models": one("SELECT COUNT(DISTINCT arch_name) FROM model_version"),
                "daily": daily[::-1], "points": points}

    def export(self):
        return [dict(r) for r in self.con.execute(
            "SELECT r.*, s.file_path, q.status AS review_status, q.reviewed_at FROM inference_result r "
            "JOIN media_sample s USING(sample_id) LEFT JOIN review_queue q USING(result_id) ORDER BY r.created_at")]
