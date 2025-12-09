# 🚗 Auto MPG Prediction – Linear Regression Project

This project builds and evaluates a **Linear Regression model** to predict a car's fuel efficiency (**MPG – miles per gallon**) using the classic **Auto MPG dataset**.  
It includes end-to-end steps from data loading, cleaning, EDA, feature engineering, model training, evaluation, and saving the trained model as a `.pkl` file.

---

## 🗂 Project Files Included

| File Name | Description |
|-----------|-------------|
| `Linear_Regression_Project2.ipynb` | Jupyter Notebook with full workflow: EDA, preprocessing, model training & evaluation |
| `LinearRegression.pkl` | Saved model using Linear Regression after outlier removal |

---

## 📊 Dataset Information

- **Name:** Auto MPG Dataset  
- **Source:** Kaggle / UCI (downloaded via `kagglehub`)  
- **Prediction Target:** `mpg` (Miles per gallon)

### **Features Used**
- `horsepower`
- `weight`
- `acceleration`

---

## 🧹 Data Preprocessing Steps
✔ Replace `?` values with NaN  
✔ Drop missing values  
✔ Convert features to numeric  
✔ Remove outliers using IQR method  
✔ Perform correlation analysis and scatterplots

---

## 📈 Model Development

### 📍 Model 1 — Before Outlier Removal
- Trained on original dataset
- Evaluated using **MSE** & **R² score**

### 📍 Model 2 — After Outlier Removal
- Cleaned dataset improves performance
- Model chosen and saved as `.pkl`

### 🔧 Saving the Model
```python
with open("LinearRegression.pkl", "wb") as f:
    pickle.dump(mode_after, f)
````

---

## 💾 Using the Saved Model

```python
import pickle
import numpy as np

with open("LinearRegression.pkl", "rb") as f:
    model = pickle.load(f)

sample = np.array([[130.0, 15.5, 3504.0]])   # horsepower, acceleration, weight
print("Predicted MPG:", model.predict(sample)[0])
```

---

## 🚀 How to Run the Project

```bash
git clone <repo-url>
cd <project-folder>
pip install -r requirements.txt
jupyter notebook Linear_Regression_Project2.ipynb
```

---

## 🔮 Future Improvements

* Add models: Lasso, Ridge, Random Forest
* Deploy using **Streamlit** or **Flask**
* Perform hyperparameter tuning
* Add more feature engineering

---

## 🏁 Conclusion

This project demonstrates a complete regression workflow from raw data to a deployment-ready model.
Feel free to improve and experiment further! 😊

````

