
# 4. Các hàm loss khác trong classification (ngoài Categorical Cross-Entropy)

## 1. Ý tưởng chính (Motivation)

Mặc dù Categorical Cross-Entropy là lựa chọn mặc định cho các mô hình phân loại hiện đại, nhiều hàm loss cổ điển từng giữ vai trò quan trọng trong lịch sử phát triển của học máy. Những hàm loss này phản ánh các quan điểm khác nhau về cách mô hình hoá khái niệm “đúng - sai”, mức độ tin cậy của dự đoán, và hành vi tối ưu hóa kỳ vọng.

Ba hàm loss kinh điển nhất - **Zero-One Loss**, **Exponential Loss**, và **Hinge Loss** - là nền tảng để hiểu tại sao Cross-Entropy (và Logistic Loss nói chung) trở thành chuẩn mực hiện nay. Chúng đại diện cho ba triết lý học khác nhau: tối ưu trực tiếp sai số, nhấn mạnh mẫu khó, và tối ưu hóa margin.

## 2. Các hàm loss

### 2.1 Zero-One Loss - mô tả trực tiếp mục tiêu classification

#### Trực giác

Đây là định nghĩa thuần túy nhất của bài toán phân loại:  
dự đoán đúng → không bị phạt,  
dự đoán sai → phạt một đơn vị.  

Zero-One Loss không xét mức độ sai ít hay sai nhiều, chỉ quan tâm đầu ra cuối cùng.

#### Công thức

$$
\mathcal{L}_{0-1}(y, \hat{y}) =
\begin{cases}
0 & \text{khi } \hat{y} = y \\
1 & \text{khi } \hat{y} \ne y
\end{cases}
$$

#### Ý nghĩa học thuật

Zero-One Loss phản ánh đúng mục tiêu classification, nhưng:

- không khả vi,  
- không cho biết hướng điều chỉnh,  
- landscape tối ưu hóa rời rạc, rất nhiều điểm kẹt.

Do đó, nó **không thể dùng để huấn luyện bằng gradient descent**.  
Toàn bộ các loss như Logistic, Hinge, Exponential đều là **surrogate** nhằm xấp xỉ Zero-One theo cách mượt hơn, tối ưu hóa được.

---

### 2.2 Exponential Loss - nhấn mạnh mạnh mẽ vào các mẫu khó (triết lý Boosting)

#### Trực giác

Exponential Loss mô phỏng hành vi:  
**một lỗi sai được đưa ra với sự “tự tin cao” sẽ bị phạt rất mạnh**.

Đây chính là cơ chế học của AdaBoost: tập trung ngày càng nhiều vào những mẫu mà mô hình đang phân loại sai.

#### Công thức

$$
\mathcal{L}_{\text{exp}}(y, f(x)) = e^{-y f(x)}
$$

Trong đó:

- $y \in \{-1, +1\}$ là nhãn nhị phân,  
- $f(x)$ là logit hoặc score của mô hình.

#### Ý nghĩa học thuật

- Nếu dự đoán đúng → $y f(x)$ lớn → hàm mũ rất nhỏ.  
- Nếu dự đoán sai → $y f(x) < 0$ → loss tăng theo cấp số nhân.

Trong AdaBoost:

- mẫu sai có loss lớn → trọng số tăng,  
- dẫn tới việc được “học kỹ hơn” ở các vòng tiếp theo.

##### Liên hệ với tối ưu hóa margin 

Exponential Loss cũng tối ưu hóa một dạng margin, nhưng *không có vùng phẳng như Hinge Loss*.  
Vì vậy mô hình có xu hướng **không bao giờ dừng tăng độ tin tưởng** trên các mẫu dự đoán đúng, khiến boosting đôi khi quá tập trung vào outlier hoặc nhiễu.

---

### 2.3 Hinge Loss - tối ưu hóa margin (tư duy SVM)

#### Trực giác

Hinge Loss không chỉ yêu cầu mô hình dự đoán đúng, mà còn dự đoán đúng **với khoảng cách an toàn (margin)** so với ranh giới phân chia.

“Đúng nhưng không chắc chắn” vẫn bị xem là chưa đạt yêu cầu.

#### Công thức nhị phân

$$
\mathcal{L}_{\text{hinge}}(y, f(x)) = \max(0,\; 1 - y f(x))
$$

#### Ý nghĩa học thuật

- Nếu $y f(x) \ge 1$: đúng và margin đủ lớn → loss = 0  
- Nếu $0 < y f(x) < 1$: đúng nhưng yếu → bị phạt tuyến tính  
- Nếu $y f(x) < 0$: sai → phạt mạnh  

Hinge Loss là nền tảng của **Support Vector Machine**, nơi mục tiêu là **tối đa hóa margin** để cải thiện khả năng tổng quát hóa.

#### Tính khả vi

Hinge Loss:

- **lồi (convex)** nhưng  
- **không khả vi tại điểm $y f(x)=1$**.

Tuy nhiên, nó **sub-differentiable**, và SGD áp dụng **subgradient** giống ReLU.

#### Multiclass Hinge Loss

Sử dụng cho phân loại $K$ lớp:

$$
\mathcal{L}(s,y) = \sum_{j \ne y} \max(0,\; s_j - s_y + 1)
$$

Đây là phiên bản thường gọi là **SVM Loss** hoặc **Crammer-Singer Loss**.

#### Liên hệ với Logistic Loss / Cross-Entropy

- Logistic Loss là một surrogate lồi, mượt (smooth), khả vi toàn phần.  
- Vì Logistic Loss ổn định hơn về mặt gradient, nó phù hợp hơn với deep learning so với Hinge Loss.  
- Cross-Entropy trong phân loại đa lớp là mở rộng tự nhiên của Logistic Loss.

---

## 3. So sánh trực quan bằng lời

- **Zero-One Loss**: dạng bước nhảy - không mượt.  
- **Hinge Loss**: tuyến tính khi sai → phẳng ngay khi đạt margin.  
- **Logistic Loss**: đường cong mượt, không có điểm gãy.  
- **Exponential Loss**: giảm rất nhanh khi đúng, tăng rất mạnh khi sai.  

### Mức độ mượt (smoothness)

$$
\text{Zero-One} \;<\; \text{Hinge} \;<\; \text{Logistic} \;<\; \text{Exponential}
$$

(Zero-One không khả vi; Hinge có điểm không khả vi; Logistic mượt hoàn toàn; Exponential cực nhạy với lỗi.)

---

## 4. Phần code

### Zero-One Loss (dùng để đánh giá, không dùng để huấn luyện)

    import torch

    # y_true: (batch,)
    # y_pred_logits: (batch, num_classes)
    def zero_one_loss(y_true, y_pred_logits):
        pred = y_pred_logits.argmax(dim=1)
        return (pred != y_true).float().mean()

### Exponential Loss

    import torch
    def exponential_loss(logits, target):
        return torch.exp(- target * logits).mean()

### Multiclass Hinge Loss

    import torch
    def multiclass_hinge_loss(scores, target):
        batch, K = scores.shape
        correct = scores[torch.arange(batch), target].unsqueeze(1)
        margins = torch.clamp(scores - correct + 1, min=0)
        margins[torch.arange(batch), target] = 0
        return margins.mean()
