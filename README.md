# Jewelry Data Analysis & Machine Learning

> **Customer Segmentation & Churn Prediction for an E-commerce Jewelry Business**

Dự án phân tích dữ liệu khách hàng cho một hệ thống thương mại điện tử kinh doanh trang sức tại Việt Nam. Project tập trung vào phân tích hành vi khách hàng, phân khúc khách hàng, dự báo giá trị khách hàng và dự đoán khả năng rời bỏ (churn) bằng Python và Machine Learning.

---

## Project Overview

Mục tiêu của project:

- Phân khúc khách hàng theo giá trị và hành vi mua sắm.
- Xác định nhóm khách hàng có giá trị cao (VIP).
- Ước tính Customer Lifetime Value (CLV).
- Dự đoán chi tiêu của khách hàng trong kỳ tiếp theo.
- Phát hiện khách hàng có nguy cơ rời bỏ.
- Đưa ra insight hỗ trợ chiến lược chăm sóc và giữ chân khách hàng.

Dataset gồm **3.593 khách hàng và 24 biến dữ liệu gốc**.

---

## Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

### Machine Learning

- K-Means Clustering
- Linear Regression
- Random Forest Classifier

---

## Dataset

Một số nhóm dữ liệu chính:

| Nhóm | Biến tiêu biểu |
|---|---|
| Customer | `customer_id`, `age`, `gender` |
| Geography | `province`, `region` |
| Product | `jewelry_category`, `material`, `karat_purity`, `weight_grams` |
| Purchase | `purchase_amount_vnd`, `previous_purchases` |
| Customer Experience | `review_rating`, `membership_status` |
| Promotion | `discount_applied`, `promo_code_used` |
| Service | `payment_method`, `shipping_type`, `purchase_channel` |
| Customer Value | `monetary_est_vnd`, `clv_est_vnd` |
| Prediction | `next_period_spend_est_vnd`, `churn`, `churn_proba` |

---

## Data Cleaning

Quy trình xử lý dữ liệu:

1. Kiểm tra Missing Values.
2. Kiểm tra Duplicate Records.
3. Phát hiện Outliers bằng IQR.
4. Kiểm tra và chuẩn hóa dữ liệu.
5. Mã hóa các biến categorical bằng One-Hot Encoding khi xây dựng mô hình.

Kết quả:

- 3.593 bản ghi.
- Không có giá trị thiếu.
- Không có bản ghi trùng lặp.
- Outlier được xem xét dựa trên bối cảnh kinh doanh.

---

## Exploratory Data Analysis

Phân tích dữ liệu tập trung vào:

- Độ tuổi khách hàng.
- Giá trị đơn hàng.
- Số lần mua trước đó.
- Điểm đánh giá.
- Mối tương quan giữa các biến.
- Giá trị đơn hàng theo danh mục trang sức.
- Giá trị đơn hàng theo dịp mua.
- Hành vi khách hàng theo khu vực, giới tính và trạng thái thành viên.

---

## Customer Dashboard

Dashboard tổng quan gồm:

### Business KPIs

- Total Customers
- Estimated Total Spending
- Average CLV
- Churn Rate

### Customer Analysis

- Churn theo khu vực.
- Chi tiêu theo giới tính.
- Phân bố Review Rating.
- Churn theo Membership Status.
- Churn theo Discount Applied.

### Product & Geography

- Phân bố khách hàng theo tỉnh/thành.
- Chi tiêu theo khu vực.
- Danh mục trang sức phổ biến.
- Chất liệu trang sức phổ biến.

---

## Customer Segmentation — K-Means

K-Means được sử dụng để phân khúc khách hàng dựa trên các biến RFM/CLV:

- `recency_proxy_days`
- `previous_purchases`
- `monetary_est_vnd`
- `clv_est_vnd`

Các biến được chuẩn hóa bằng `StandardScaler`.

Số lượng cluster được lựa chọn bằng Elbow Method với **K = 4**.

### Customer Segments

| Segment | Customers | Percentage |
|---|---:|---:|
| VIP Customers | 399 | 11.1% |
| Loyal Customers | 1,233 | 34.3% |
| At-Risk Customers | 631 | 17.6% |
| New / Regular Customers | 1,330 | 37.0% |

---

## Customer Lifetime Value Prediction

Sử dụng **Linear Regression** để dự đoán `clv_est_vnd`.

### Performance

- **R² ≈ 0.808**
- **MAE ≈ 13.65 triệu VND**

Mô hình giải thích khoảng 80,8% biến động CLV trên tập kiểm tra.

---

## Next Period Spending Prediction

Sử dụng **Linear Regression** để dự đoán:

`next_period_spend_est_vnd`

### Performance

- **R² ≈ 0.252**
- **MAE ≈ 8.28 triệu VND**

---

## Customer Churn Prediction

Nhãn `churn` được xây dựng dựa trên các tín hiệu hành vi như:

- Tần suất mua thấp.
- Không đăng ký thành viên.
- Review Rating thấp.
- Không sử dụng promo code.
- Không áp dụng discount.

Sau đó sử dụng **Random Forest Classifier** để dự đoán khả năng churn.

### Performance

| Metric | Score |
|---|---:|
| Accuracy | ~72.5% |
| Precision | ~47.6% |
| Recall | ~58.3% |
| F1-score | ~52.4% |

---

## Key Business Insights

### VIP Customers

Nhóm VIP chiếm khoảng **11,1%** khách hàng nhưng có CLV trung bình cao nhất.

→ Có thể ưu tiên chương trình loyalty, ưu đãi cá nhân hóa và chăm sóc riêng.

### Churn Risk

Tỷ lệ churn toàn bộ dataset khoảng **26,0%**.

→ Có thể tập trung các chiến dịch giữ chân vào nhóm khách hàng có dấu hiệu rời bỏ.

### Customer Segmentation

4 nhóm khách hàng giúp doanh nghiệp xây dựng chiến lược marketing và chăm sóc phù hợp với từng nhóm.

---

## Analysis Workflow

```text
Raw Customer Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Exploratory Data Analysis
        ↓
Dashboard & Visualization
        ↓
 ┌───────────────┬──────────────────┬───────────────────┐
 ↓               ↓                  ↓
K-Means       Linear Regression   Random Forest
Clustering       Prediction        Classification
 ↓               ↓                  ↓
Customer         CLV & Next       Churn Prediction
Segments         Spending
 └───────────────┴──────────────────┴───────────────────┘
                    ↓
             Business Insights
```

---


---

## How to Run

### Option 1 — Run with Google Colab (Recommended)

This project was developed as a **Google Colab Notebook**, so Google Colab is the easiest way to run it.

#### Step 1 — Open the Notebook

Download or open:

`Jewelry_Data_Analysis_ML.ipynb`

You can also upload the `.ipynb` file directly to Google Colab.

#### Step 2 — Upload the Dataset

When the notebook reaches the **Data Loading & Data Cleaning** section, it uses:

```python
from google.colab import files
uploaded = files.upload()
```

Select the dataset:

```text
trang_suc_processed_full.csv
```

The notebook then loads the dataset using:

```python
CSV_PATH = "trang_suc_processed_full.csv"
df = pd.read_csv(CSV_PATH)
```

> **Important:** If Google Colab changes the uploaded filename to something like `trang_suc_processed_full (4).csv`, either rename the file back to `trang_suc_processed_full.csv` or update `CSV_PATH` to the actual filename.

#### Step 3 — Run the Notebook

Run the cells from **top to bottom**.

You can use:

**Runtime → Run all**

or click the ▶️ button on each cell in order.

The notebook will perform:

```text
Import Libraries
      ↓
Upload Dataset
      ↓
Load & Clean Data
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Dashboard
      ↓
K-Means Clustering
      ↓
Linear Regression
      ↓
Random Forest Classification
      ↓
Model Evaluation
      ↓
Business Insights
```

### Option 2 — Run Locally with Jupyter Notebook

#### 1. Install Python

Recommended: **Python 3.10+**

#### 2. Install required libraries

Open Terminal / Command Prompt and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

#### 3. Download the project

Clone the repository:

```bash
git clone https://github.com/yennhi712/jewelry-data-analysis.git
cd jewelry-data-analysis
```

Or download the repository as a ZIP file from GitHub and extract it.

#### 4. Make sure the files are in the same folder

```text
jewelry-data-analysis/
│
├── Jewelry_Data_Analysis_ML.ipynb
├── trang_suc_processed_full.csv
└── README.md
```

#### 5. Start Jupyter Notebook

Run:

```bash
jupyter notebook
```

Open:

```text
Jewelry_Data_Analysis_ML.ipynb
```

Then run the notebook from top to bottom.

> **Note:** The notebook currently contains `google.colab.files.upload()`, which is designed for Google Colab. If running locally, the simplest approach is to keep `trang_suc_processed_full.csv` in the same folder as the notebook and use:
>
> ```python
> CSV_PATH = "trang_suc_processed_full.csv"
> df = pd.read_csv(CSV_PATH)
> ```
>
> You may skip the Colab upload cell when running locally.

### Required Python Packages

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

##  Project Structure

```text
jewelry-data-analysis/
│
├── Jewelry_Data_Analysis_ML.ipynb
├── trang_suc_processed_full.csv
└── README.md
```

### Files

- `Jewelry_Data_Analysis_ML.ipynb` — Notebook chứa toàn bộ quy trình phân tích và Machine Learning.
- `trang_suc_processed_full.csv` — Dataset được sử dụng trong project.
- `README.md` — Project documentation.

---

## Limitations

Dataset là dạng snapshot, không phải transaction history theo thời gian.

Vì vậy:

- Recency không phải dữ liệu giao dịch thực tế theo lịch sử.
- Monetary và CLV có thành phần ước tính.
- Next-period spending được ước tính.
- Churn được xây dựng theo business rules.

Do đó, kết quả chủ yếu mang tính phân tích và minh họa phương pháp, chưa phải hệ thống dự báo production-ready.

---

## Future Improvements

- Bổ sung transaction history theo thời gian.
- Tính Recency và CLV từ dữ liệu giao dịch thực tế.
- Thử nghiệm thêm DBSCAN và Hierarchical Clustering.
- So sánh thêm XGBoost/LightGBM.
- Tối ưu threshold cho bài toán Churn.
- Xây dựng hệ thống churn scoring tích hợp CRM.
- Phát triển dashboard tương tác bằng Power BI hoặc Streamlit.

---

##  Conclusion

Project xây dựng một quy trình **Customer Analytics & Machine Learning end-to-end**, từ Data Cleaning, EDA, Visualization đến Customer Segmentation, CLV Prediction, Next-period Spending Prediction và Churn Prediction.

---

 **Thank you for visiting this project!**
