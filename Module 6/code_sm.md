# Hướng Dẫn Cài Đặt Softmax Regression Từ Zero Bằng Python

Trong bài viết này, chúng ta sẽ tự xây dựng **Softmax Regression** — mô hình phân loại đa lớp — hoàn toàn bằng **NumPy**.

Mọi khái niệm đều dựa trên tài liệu **Softmax Regression** bạn đã tải lên.

Bài viết sẽ giúp bạn:

- Hiểu *one-hot encoding*  
- Tự xây *Softmax function*  
- Tính *cross-entropy loss*  
- Tính *gradient* bằng công thức từ tài liệu  
- Cập nhật tham số bằng *gradient descent*  


![Screenshot 2025-11-24 203640.png](/static/uploads/20251124_203704_7ceff515.png)

# Giai đoạn Huấn luyện (Training Phase)

1. **Khởi tạo trọng số (Initialize weights)**  
   Bắt đầu bằng cách gán giá trị ban đầu cho các trọng số và bias. Thường dùng giá trị ngẫu nhiên hoặc theo một phương pháp chuẩn hóa.

2. **Chọn một mẫu (x, y) từ dữ liệu huấn luyện (Pick a sample)**  
   Lấy từng cặp dữ liệu đầu vào và nhãn tương ứng để tính toán.

3. **Tính đầu ra dự đoán $( \hat{y} )$ (Compute output)**  
   Dùng trọng số hiện tại để dự đoán đầu ra từ mẫu dữ liệu. Đây là bước forward propagation.

4. **Tính loss (Compute loss)**  
   So sánh giá trị dự đoán $( \hat{y} )$ với giá trị thật y để tính hàm mất mát (loss).

5. **Tính đạo hàm (Compute derivative)**  
   Tính gradient của loss theo các trọng số. Đây là bước quan trọng để biết cần điều chỉnh trọng số như thế nào.

6. **Cập nhật tham số (Update parameters)**  
   Sử dụng gradient và learning rate để cập nhật trọng số, nhằm giảm loss.

7. **Lặp lại từ bước 2 cho mẫu tiếp theo (Repeat from step 2)**  
   Tiếp tục với các mẫu khác cho đến khi toàn bộ dữ liệu được huấn luyện hoặc đạt điều kiện dừng.

💡 **Chú thích thêm:** Quá trình này lặp đi lặp lại nhiều lần (epochs) trên toàn bộ tập dữ liệu để mô hình học được các đặc trưng và giảm lỗi dự đoán.


## B1: Chuẩn bị dữ liệu và One-hot Encoding

Tài liệu chỉ ra rằng cross-entropy nhiều lớp được viết gọn bằng one-hot:

$$
L = -\sum_{j=1}^{k} y_j \log(\hat{y}_j)
$$

Nên ta cần chuyển nhãn (0, 1, …, k−1) → vector one-hot.

###### 1.1 Code one-hot encoding:

```python
def convert_one_hot(y, k):  
    one_hot = np.zeros((len(y), k))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot
```

###### 1.2 Thêm cột Intercept

Softmax Regression dùng:

$$
z = \theta^T x = 
\begin{bmatrix}
b \\
w
\end{bmatrix}^T
\begin{bmatrix}
1 \\
x
\end{bmatrix}
$$

Nên ta thêm 1 cột toàn số 1 vào ma trận `X`:

```python
  intercept = np.ones((X.shape[0], 1))
  X = np.concatenate((intercept, X), axis=1)
```

### B2: Khởi tạo tham số θ

Nếu số chiều của input = 1 → θ có shape (2 × k).

Ví dụ:

```python
theta = np.array([[0.1, 0.05], 
                  [0.2, -0.1]])
```

### B3: Vòng lặp huấn luyện Softmax Regression

Huấn luyện dựa theo công thức gradient trong tài liệu:

**Forward:**

$$
z = \theta^T x
$$

$$
\hat{y} = \text{softmax}(z)
$$

**Loss (cross entropy):**

$$
L = -y^T \log(\hat{y})
$$

**Gradient:**

$$
\frac{\partial L}{\partial \theta} = x(\hat{y} - y)^T
$$

🔥 **Toàn bộ vòng lặp training:**

```python
learning_rate = 0.1
losses = []
max_epoch = 1

for epoch in range(max_epoch):
    for i in range(N): 
        xi = X[i]
        yi = y_one_hot[i]
        
        # reshape to column vectors
        xi = xi.reshape((2,1))
        yi = yi.reshape((2,1))
        
        # compute z
        z = theta.T.dot(xi)        
        
        # compute y_hat (softmax)
        exp_z = np.exp(z)
        y_hat = exp_z / np.sum(exp_z, axis=0)
        
        # compute loss
        loss = -yi.T.dot(np.log(y_hat))
        losses.append(loss[0])
        
        # compute gradient
        dz = y_hat - yi              # (2×1)
        dtheta = xi.dot(dz.T)        # (2×2)
        
        # update parameters
        theta = theta - learning_rate * dtheta
```


### Giải thích từng bước

✔ **Forward pass**  
Ta tính $ z = \theta^T x $ và softmax $(\hat{y})$.

✔ **Loss**  
Dựa đúng công thức trong tài liệu:

$$
L = -y^T \log(\hat{y})
$$

Vì \(y\) là one-hot → chỉ lấy log(p) của class đúng.

✔ **Gradient**  
Tài liệu chứng minh:

$$
\frac{\partial L}{\partial z} = \hat{y} - y
$$

Từ chain rule:

$$
\frac{\partial L}{\partial \theta} = x (\hat{y} - y)^T
$$

✔ **Update**  

$$
\theta := \theta - \eta \frac{\partial L}{\partial \theta}
$$





