# Hồi quy Softmax (Softmax Regression)

## 2.1. Ý tưởng chính
- Hồi quy softmax (softmax regression) tổng quát hóa mô hình hồi quy logistic (logistic regression) để tối ưu hóa lời giải cho các bài toán phân loại đa lớp. Thay vì sử dụng hàm sigmoid như trong hồi quy logistic, mô hình hồi quy softmax biến các điểm số thô (hay còn gọi là điểm logit) thành một phân phối xác suất rõ ràng trên tất cả các lớp. 

- *Vậy các điểm số thô / điểm logit này đến từ đâu?* Như trong hồi quy logistic, điểm logit là một kết quả của 1 tổ hợp có trọng số của các đặc trưng đầu vào với hệ số nghiêng (bias). Mỗi logit trong hồi quy softmax biểu diễn mức độ ảnh hưởng của các đặc trưng (features) lên một lớp/nhãn (class/label) cụ thể cho từng mẫu (sample). 
 
- *Tại sao một logit chỉ ảnh hưởng tới một mẫu nhất định?* Hồi quy logistic chỉ có một bộ tham số duy nhất, nên mỗi mẫu chỉ có một logit để tạo ra chính xác một xác suất. Trong khi đó, mỗi lớp trong hồi quy softmax có riêng một bộ tham số để tính toán một điểm sô thô cho chính nó, từ đó hình thành phân phối xác suất trên nhiều lớp.

- *Hàm softmax đóng vai trò gì trong mô hình này?* Hàm softmax nhận điểm số thô để cho ra xác suất của các lớp sao cho tổng của tất cả xác suất trên một mẫu bằng $1$. Những xác suất này mô tả sự tin tưởng của mô hình dành cho từng lớp. Lớp được dự đoán chính là lớp có xác suất lớn nhất. 

## 2.2. Công thức
Cho một tập hợp điểm dữ liệu với $N$ đặc trưng và $m$ lớp. Chúng ta sẽ vectơ hóa dữ liệu đầu vào và đầu ra để thuận tiện ký hiệu, tính toán, và lập trình. 

### 2.2.1. Điểm số thô - Logits
Như vậy, ta cơ bản đã hiểu ý tưởng của các điểm số thô và vai trò của chúng trong hồi quy softmax. Vậy logits được thực sự đưa vào mô hình như thế nào? Chúng ta sẽ cùng xây biểu thức toán học cho logits theo từng bước cụ thể như sau. 

- **Một mẫu, một lớp:** 

    Với $j \in \{1, \cdots, N\}$ và $i \in \{0, \cdots, m - 1\}$, ta có 
    - $x^{(k)}_j$ là đặc trưng đầu vào thứ $j$ của mẫu $k$,
    - $w_{ij}$ là tham số đặc trưng thứ $j$ cửa lớp $i$,
    - $w_{i0}$ là hệ số nghiêng của lớp $i$,
    - $\mathbf{x}^{(k)} = \begin{pmatrix} 1 \\ x^{(k)}_1 \\ \cdots \\ x^{(k)}_n \end{pmatrix} \in \mathbb{R}^{n+1}$ là vectơ đầu vào kèm hệ số nghiêng của mẫu $k$, 
    - $\mathbf{\theta}_i = \begin{pmatrix} w_{i0} \\ w_{i1} \\ \cdots \\ w_{in} \end{pmatrix} \in \mathbb{R}^{n+1}$ là vectơ tham số kèm hệ số nghiêng của lớp $i$, và
    - $z^{(k)}_i$ là logit của lớp $i$.

    Phương trình tuyến tính biến vectơ mẫu $\mathbf{x}^{(k)}$ thành một logit của lớp $i$ như sau

    $$
    z^{(k)}_i = w_{i1} x^{(k)}_1 + \cdots + w_{in} x^{(k)}_n + w_{i0} = \sum_{j=0}^{n} w_{ij}x^{(k)}_j = \mathbf{\theta}_i^T \mathbf{x}^{(k)}.
    $$

- **Một mẫu, $m$ lớp:** 
    
    Ta sẽ sử dụng lại vectơ mẫu $\mathbf{x}^{(k)} \in \mathbb{R}^{n+1}$ ở trên. Với mẫu này, ta cần một vectơ lưu trữ các logits của tất cả các lớp và một ma trận lưu trữ các bộ trọng số. Như vậy, ta có
    
    - $\mathbf{z}^{(k)} \in \mathbb{R}^{m \times 1}$ là vectơ logit trong đó mỗi thành phần là $z^{(k)}_i$, và
    - $\Theta \in \mathbb{R}^{(n+1) \times m}$ là ma trận trọng số có hệ số nghiêng trong đó mỗi cột là $\mathbf{\theta}_i$. 

    Khi đó, các điểm số thô cho tất cả các lớp của mẫu này là

    $$
    \mathbf{z}^{(k)} = \Theta^T \mathbf{x}^{(k)}.
    $$

- **$K$ mẫu, $m$ lớp:**

    Ta sẽ sử dụng lại ma trận trọng số $\Theta \in \mathbb{R}^{(n+1) \times m}$. Ngoài ra, ta cần một ma trận lưu trữ các mẫu và một ma trận khác lưu trữ logits tương ứng. Ta có

    - $X \in \mathbb{R}^{K \times (n+1)}$ là ma trận mẫu trong đó mỗi dòng biểu diễn một mẫu $\mathbf{x}^{(k)} \in \mathbb{R}^{n+1}$, và 
    - $Z \in \mathbb{R}^{K \times m}$ là ma trận logit trong đó mỗi thành phần là logit của mẫu và lớp tương ứng.

    Khi đó, ma trận logit cho $K$ mẫu và $m$ lớp là

    $$
    Z = X \Theta.
    $$

### 2.2.2. Softmax Function
- *Tại sao chỉ dựa vào logit thôi là chưa đủ?* Mặc dù các điểm số thô phản ánh mức độ ảnh hưởng của các đặc trưng đầu vào lên một lớp cụ thể nhưng chúng không thể hiện được mối tương quan giữa các lớp với nhau. Do đó, ta cần biến các logit thành xác suất để đạt được mục đích ban đầu. 

- *Tại sao cần hàm softmax?* Các logit là các số thực, có thể âm, dương, hoặc bằng $0$. Tuy nhiên, số âm không thể tự biểu thị được một xác suất theo cách thông thường. Chính vì thế, ta nên dùng hàm softmax để đảm bảo tính không âm. Hàm softmax có thể biến điểm số thô thành các xác suất có ý nghĩa với tổng bằng $1$. Kết quả thể hiện độ mạnh của lớp tương ứng so với các lớp khác.

- **Hàm softmax:**
    $$
    softmax(z) = \frac{e^z}{\sum_c e^{z_c}}
    $$
- **Hàm softmax trong hồi quy softmax:**
    Cho mẫu $\mathbf{x}^{(k)}$ với vectơ logit $\mathbf{z}^{(k)}$, hàm softmax tính xác suất của lớp $i$ như sau  

    $$
    \displaystyle \hat{y_i}^{(k)} = p(y=i | \mathbf{z}^{(k)}) = \frac{e^{{z^{(k)}_i}}}{\sum_{c=0}^{m-1}e^{z^{(k)}_c}}.
    $$

    Công thức này thỏa mãn các yêu cầu:  
    - tất cả các xác suất đều không âm nhờ hàm mũ,
    - tổng của tất cả các xác suất cho một mẫu bằng $1$, và 
    - xác suất và logit tỉ lệ thuận với nhau.

- **Ví dụ:**
    Xét một mẫu với điểm số thô $\mathbf{z} = \begin{pmatrix} z_0 \\ z_1 \\ z_2 \end{pmatrix} = \begin{pmatrix} 2.0 \\ 1.0 \\ 0.1 \end{pmatrix}$. Áp dụng hàm mũ cho mỗi logit, ta có 
    
    $$
    e^{2.0} = 7.3891, \quad e^{1.0} = 2.7183, \quad e^{0.1} = 1.1052,
    $$
    và
    $$
    \sum_{c = 0}^{2} e^{z_c} = e^{2.0} + e^{1.0} + e^{0.1} = 7.3891 + 2.7183 + 1.1052 = 11.2126.
    $$

    Xác suất của lớp $0$ được dự đoán là 

    $$
    \hat{y_0} = p(y = 0 | \mathbf{z}) = \frac{e^{2.0}}{\sum_{c = 0}^{2} e^{z_c}} = \frac{7.3891}{11.2126} \approx 0.6590.
    $$

    Tính tương tự, ta có vectơ xuất cho lớp $i$ là 

    $$
    \hat{y} = \begin{pmatrix} \hat{y_0} \\ \hat{y_1} \\ \hat{y_2} \end{pmatrix} \approx \begin{pmatrix} 0.6590 \\ 0.2424 \\ 0.0986 \end{pmatrix}
    $$

## 2.3. Quy trình huấn luyện mô hình hồi quy softmax
Mục tiêu của mô hình hồi quy softmax là điều chỉnh tham số sao cho xác suất được dự đoán chính xác nhất có thể. 

- **Forward propagation:**
    - Bước 1: Tính logit bằng phép biến đổi tuyến tính.
    - Bước 2: Áp dụng hàm softmax để chuyển logit thành phân phối xác suất. Lớp có xác suất cao nhất là lớp dự đoán.

- **Backward propagation:** (chi tiết ở phần sau)
    - Bước 3: Tính hàm mất mát (cross-entropy loss) để kiểm tra độ hợp giữa nhãn dự đoán và nhãn thật.
    - Bước 4: Tính đạo hàm (gradient) giúp xác định các tham số cần thay đổi như thế nào để giảm hàm mất mát.
    - Bước 5: Cập nhật tham số bằng gradient descent.

Ta lặp lại quá trình này cho các epoch tới khi hàm mất mát hội tụ và dự đoán ổn định và chính xác.

## 2.4. Ghi chú
Trước khi kết thúc phần này, hãy cùng điểm lại một vài lưu ý quan trọng về mô hình hồi quy softmax.

### 2.4.1. Ưu điểm
- Mô hình dễ hiểu và dễ diễn giải kết quả.
- Mô hình được triển khai và huấn luyện hiệu quả.
- Hàm hàm mất mát là hàm lồi (convex) nên đảm bảo tìm được cực tiểu toàn cục (the global minimum) mà không gặp vấn đề về cục tiểu cục bộ (local minima).

### 2.4.2. Nhược điểm
- **Tràn số:**
    Khi logit quá lớn, số mũ $e^{z_i}$ có thể vượt quá giá trị lớn nhất mà kiểu dữ liệu có thể chứa được. Nếu không được giải quyết triệt để thì có thể dẫn đến tình trạng sai số. Để tránh tình trạng này, ta có thể giới hạn số mũ bằng $m = \underset{j}{max} z_i$ mà vẫn giữ nguyên kết quả. Vậy, ta có

    $$
    \hat{y_i} = \frac{e^{z_i-m}}{\sum_c e^{z_c-m}}
    $$

    Ta thấy trừ đi một hằng số trên mũ (logits) không làm thay đổi kết quả của hàm softmax vì $e^m$ bị khử trong phân số. 

- **Quan hệ phi tuyến:**
    Hồi quy softmax là mô hình một tầng/lớp với hàm giả thuyết là phép biến đổi tuyến tính. Tính chất này khiến cho mô hình nhạy cảm với các ngoại lệ (outliers) cũng như khó xử lý mối quan hệ phi tuyến tính một cách hiệu quả. Để giải quyết vấn đề này, ta cần có dạng mạng nhiều tầng như Multilayer Perceptron (MLP).

<!-- # 2. Softmax Regression 

## 2.1. Main Idea
- Softmax Regression, which handles multi-class classification problems, is a generalization of logistic regression. Instead of using the sigmoid function, softmax regression employs the softmax function to turn raw scores into an explicit probability distribution across all classes. 

- *Where do these raw scores (also known as logits) come from?* Like in logistic regression, a logit results from a linear combination which is a weighted sum of input features and a bias. Each of them in softmax regression represents the features' influence on a particular class. 

- *Why does one logit only support a specific class?* In logistic regression, each sample produces just one probability as there is only one score derived from a single set of parameters. In contrast, each class in softmax regression has its own set of parameters to create one raw score thereby generate a probability distribution over multiple classes. 

- *How does the softmax function contribute to the model?* A raw score is fed to the softmax function to compute probabilities over all the classes for one sample. Such probabilities sum to $1$, which describes how strongly the model favors one class over the rest. Thus, the predicted class is simply the one with the highest probability within this distribution.

## 2.2. Formula
Consider a dataset of $N$ features and $m$ classes. For mathematical convenience and more efficient computation, we'll vectorize input and output.

### 2.2.1. Raw Scores (Logits)
- We understand the idea of raw scores and their contribution to softmax regression. The question now is how these logits technically fit into the model. Let's formalize the mathematical defintion of logits.

- **One sample, one particular class:** 

    For $j \in \{1, \cdots, N\}$ and $i \in \{0, \cdots, m - 1\}$, let 
    - $x^{(k)}_j$ be the $k^{th}$ sample's $j^{th}$ input feature,
    - $w_{ij}$ be the $j^{th}$ feature's parameter for class $i$,
    - $w_{i0}$ be the bias term for class $i$,
    - $\mathbf{x}^{(k)} = \begin{pmatrix} 1 \\ x^{(k)}_1 \\ \cdots \\ x^{(k)}_n \end{pmatrix} \in \mathbb{R}^{n+1}$ be the $k^{th}$ sample's input vector with bias included, 
    - $\mathbf{\theta}_i = \begin{pmatrix} w_{i0} \\ w_{i1} \\ \cdots \\ w_{in} \end{pmatrix} \in \mathbb{R}^{n+1}$ be the weight-with-bias vector for class $i$, and
    - $z^{(k)}_i$ be the raw score (logit) for class $i$.

    A linear equation transforms the sample vector $\mathbf{x}^{(k)}$ into one raw score (logit) for class $i$ as follows.

    $$
    z^{(k)}_i = w_{i1} x^{(k)}_1 + \cdots + w_{in} x^{(k)}_n + w_{i0} = \sum_{j=0}^{n} w_{ij}x^{(k)}_j = \mathbf{\theta}_i^T \mathbf{x}^{(k)}.
    $$

- **One sample, $m$ classes:** 
    
    We'll use the same sample vector $\mathbf{x}^{(k)} \in \mathbb{R}^{n+1}$ as above. We'll introduce a vector for storing logits for all classes and a matrix for storing all sets of parameters. Consider
    
    - $\mathbf{z}^{(k)} \in \mathbb{R}^{m \times 1}$ is the logit vector where each entry is $z^{(k)}_i$, and
    - $\Theta \in \mathbb{R}^{(n+1) \times m}$ is the weight-with-bias matrix where each column is $\mathbf{\theta}_i$. 

    The logits for all classes are then given by

    $$
    \mathbf{z}^{(k)} = \Theta^T \mathbf{x}^{(k)}.
    $$

- **$K$ samples, $m$ classes:**

    We'll use the same parameter matrix $\Theta \in \mathbb{R}^{(n+1) \times m}$. We'll need a matrix to store all interested samples and another to store their logits. Suppose

    - $X \in \mathbb{R}^{K \times (n+1)}$ is the sample matrix where each row represents a sample $\mathbf{x}^{(k)} \in \mathbb{R}^{n+1}$, and 
    - $Z \in \mathbb{R}^{K \times m}$ is a logit matrix where each entry contains the logit of the corresponding sample-class pair.

    Then, the logit matrix for $K$ samples and $m$ classes is defined as

    $$
    Z = X \Theta.
    $$

### 2.2.2. Softmax Function
- *Why are raw scores not enough?* Though raw scores reflect how much input features support a specific class, they do not show where this class stands relative to the others. So converting logits into probabilities is necessary. 

- *Why do we need the softmax function?* Notice that logits can be negative, so they cannot represent valid probabilities themselves. This is why we use the softmax function, which is a method that can transform raw scores into meaningful probabilities and sum to $1$. The results express each class's strength in comparison to the rest. 

- **General softmax function:**
    $$
    softmax(z) = \frac{e^z}{\sum_c e^{z_c}}
    $$
- **Softmax function in softmax regression:**
    For a sample $\mathbf{x}^{(k)}$ with logits $\mathbf{z}^{(k)}$, the softmax function computes the probability of class $i$ as  

    $$
    \displaystyle \hat{y_i}^{(k)} = p(y=i | \mathbf{z}^{(k)}) = \frac{e^{{z^{(k)}_i}}}{\sum_{c=0}^{m-1}e^{z^{(k)}_c}}.
    $$

    This formula ensures: 
    - all the probabilities are non-negative via the exponential function,
    - all probabilities for one sample sum to $1$, and 
    - higher logits yield higher probabilities.

- **Example:**
    Suppose a sample has raw scores $\mathbf{z} = \begin{pmatrix} z_0 \\ z_1 \\ z_2 \end{pmatrix} = \begin{pmatrix} 2.0 \\ 1.0 \\ 0.1 \end{pmatrix}$. Applying the exponential function on each logit, we get

    $$
    e^{2.0} = 7.3891, \quad e^{1.0} = 2.7183, \quad e^{0.1} = 1.1052,
    $$
    and
    $$
    \sum_{c = 0}^{2} e^{z_c} = e^{2.0} + e^{1.0} + e^{0.1} = 7.3891 + 2.7183 + 1.1052 = 11.2126.
    $$

    The estimated probability for class $0$ is 

    $$
    \hat{y_0} = p(y = 0 | \mathbf{z}) = \frac{e^{2.0}}{\sum_{c = 0}^{2} e^{z_c}} = \frac{7.3891}{11.2126} \approx 0.6590.
    $$

    Thus, the estimated probability vector for the sample is 

    $$
    \hat{y} = \begin{pmatrix} \hat{y_0} \\ \hat{y_1} \\ \hat{y_2} \end{pmatrix} \approx \begin{pmatrix} 0.6590 \\ 0.2424 \\ 0.0986 \end{pmatrix}
    $$

## 2.3. Training
Now that concept is in place, let's shift our attention to how softmax regression is trained. The goal is to adjust the parameters so that the probabilities are estimated as accurately as possible.

- **Forward propagation:**
    - *Step 1:* Compute logits using the linear transformation.
    - *Step 2:* Apply the softmax function to obtain a probability distribution from logits. Those with the highest probabilities are the predicted classes.

- **Backward propagation:** (covered in later sections)
    - *Step 3:* Compute loss to measure how well the predictions match the true labels.
    - *Step 4:* Compute gradients that determine how parameters should change to reduce the loss.
    - *Step 5:* Update parameters using gradient descent to better predictions.

This process repeats over multiple epochs until the loss converges and specifically the predictions become consistently accurate.

## 2.4. Notes
Before we wrap up this section, here are a few key notes about the softmax regression model. 

### 2.4.1. Advantages
- It is easy to understand and interpret the outcomes.
- It is efficient to implement and train.
- The loss function is convex, which leaves the global mimum but local minima.

### 2.4.2. Disadvantages
- **Numerical overflow:**
    When logits are too large, the exponential $e^{z_i}$ can overflow, which leads to computational inaccuracy if not carefully handled. To avoid this situation, we can limit the exponent by $m = \underset{j}{max} z_i$ and keep the result the same. Therefore,

    $$
    \hat{y_i} = \frac{e^{z_i-m}}{\sum_c e^{z_c-m}}.
    $$

    Note that subtracting a constant $m$ from exponents (logits) does not change the result of the softmax function as $e^m$ is cancelled out.

- **Non-linear relationship:**
    Softmax regression is a single-layered model with the hypothesis function being a linear transformation. Thus, it not only can be sensitive to outliers but struggles handling non-linear relationship efficiently as well. This is when Multilayer Perceptron (MLP) comes into play. 
 -->
 
