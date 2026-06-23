Setuju. Bahkan kalau melihat screenshot jobsheet yang kamu upload, memang ada dropdown:

```python
model_names = ['Decision Tree', 'SVC']
```

dan user bisa memilih model sebelum melakukan prediksi. Jadi saat ini implementasi kita sebenarnya **lebih sederhana daripada referensi dosen**.

Sebelum deployment, sebaiknya kita revisi agar lebih mirip jobsheet.

---

# Target Revisi

Kita akan menghasilkan:

```text
model_dt.pkl
model_svc.pkl
scaler.pkl
```

dan di web:

```text
Model:
[ Decision Tree ▼ ]
[ SVC ▼ ]
```

---

# Tahap 1 — Revisi Notebook Training

Buka `model.ipynb`.

Import tambahan:

```python
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
```

---

## Training Decision Tree

```python
dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(
    X_train_scaled,
    y_train
)
```

---

## Training SVC

```python
svc_model = SVC(
    kernel='rbf',
    probability=True,
    random_state=42
)

svc_model.fit(
    X_train_scaled,
    y_train
)
```

---

## Evaluasi

Tambahkan:

```python
from sklearn.metrics import accuracy_score
```

Decision Tree:

```python
dt_pred = dt_model.predict(
    X_test_scaled
)

print(
    "Decision Tree:",
    accuracy_score(
        y_test,
        dt_pred
    )
)
```

SVC:

```python
svc_pred = svc_model.predict(
    X_test_scaled
)

print(
    "SVC:",
    accuracy_score(
        y_test,
        svc_pred
    )
)
```

Biasanya hasilnya kira-kira:

```text
Decision Tree : 0.74
SVC           : 0.81
```

(SVC sering menang pada dataset diabetes)

---

# Tahap 2 — Simpan Kedua Model

Ganti kode pickle lama menjadi:

```python
import pickle
```

Decision Tree:

```python
pickle.dump(
    dt_model,
    open(
        "model_dt.pkl",
        "wb"
    )
)
```

SVC:

```python
pickle.dump(
    svc_model,
    open(
        "model_svc.pkl",
        "wb"
    )
)
```

Scaler:

```python
pickle.dump(
    scaler,
    open(
        "scaler.pkl",
        "wb"
    )
)
```

---

# Tahap 3 — Hapus Model Lama

Nanti folder menjadi:

```text
flask-ilham
│
├── model_dt.pkl
├── model_svc.pkl
├── scaler.pkl
```

File:

```text
model.pkl
```

sudah tidak dipakai.

---

# Tahap 4 — Revisi app.py

Load kedua model.

Ganti:

```python
model = pickle.load(...)
```

menjadi:

```python
dt_model = pickle.load(
    open(
        "model_dt.pkl",
        "rb"
    )
)

svc_model = pickle.load(
    open(
        "model_svc.pkl",
        "rb"
    )
)

scaler = pickle.load(
    open(
        "scaler.pkl",
        "rb"
    )
)
```

---

Tambahkan daftar model:

```python
model_names = [
    "Decision Tree",
    "SVC"
]
```

---

## Route Home

Ganti:

```python
return render_template(
    "index.html"
)
```

menjadi:

```python
return render_template(
    "index.html",
    model_names=model_names
)
```

---

## Route Predict

Setelah membuat DataFrame:

```python
selected_model = request.form["model"]
```

---

Pilih model:

```python
if selected_model == "Decision Tree":
    prediction = dt_model.predict(
        scaled
    )[0]
else:
    prediction = svc_model.predict(
        scaled
    )[0]
```

---

Hasil:

```python
prediction_text = (
    "Diabetic"
    if prediction == 1
    else "Non-Diabetic"
)
```

---

Return:

```python
return render_template(
    "index.html",
    prediction=prediction_text,
    model_names=model_names
)
```

---

# Tahap 5 — Revisi index.html

Tambahkan dropdown sebelum field pertama.

Di dalam form:

```html
<label>Model</label>

<select name="model" required>
  {% for m in model_names %}

  <option value="{{ m }}">{{ m }}</option>

  {% endfor %}
</select>

<br /><br />
```

---

# Hasil Akhir

Web akan menjadi:

```text
Diabetes Prediction

Model:
[ Decision Tree ▼ ]

Pregnancies
Glucose
BloodPressure
...
```

atau:

```text
Model:
[ SVC ▼ ]
```

seperti referensi jobsheet.

---

# Bonus (Supaya Lebih Mirip Jobsheet)

Di screenshot jobsheet terlihat:

```python
model_names = [
    'Decision Tree',
    'SVC'
]
```

Jadi setelah revisi ini, implementasimu akan hampir identik dengan contoh dosen, hanya nama file dan repository yang berbeda.

---

Sebelum lanjut ke deployment, jalankan notebook lagi dan kirim hasil akurasi:

```text
Decision Tree: ?
SVC: ?
```

Aku ingin melihat model mana yang performanya lebih baik pada `diabetes.csv` milikmu. Biasanya SVC unggul, tapi kita bisa verifikasi dulu sebelum deploy.
