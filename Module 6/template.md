# Softmax Regression

## 1. Ý tưởng chính (Main Idea) hoặc lí do vì sao có phần kiến thức này (Motivation)

Tóm tắt lại ý chính hoặc nêu động lực bài toán

## 2. Nội dung chi tiết

Nội dung chi tiết (có thể heading 2 thành nhiều chủ đề liên tục)

### Một chủ đề con quan trọng

* Liệt kê không có thứ tự
* Có thể dùng **in đậm** (dùng `**từ khóa**`) những thứ cần nhớ.
* Hoặc *in nghiêng* (dùng `*từ khóa*`) để nhấn mạnh.

### Một chủ đề con khác (ví dụ: các bước)

1.  Dùng khi cần liệt kê có thứ tự
2.  Dùng in đậm hoặc in nghiêng nhấn mạnh

---

## 3. Công thức toán học (latex)

arkdown dùng cú pháp LaTeX để hiển thị công thức toán học.

### Công thức ngay trong dòng (Inline)

Dùng một cặp dấu đô-la `$` để bọc công thức và sẽ hiển thị ngay trong dòng văn bản

* Ví dụ: Phương trình bậc hai có dạng $ax^2 + bx + c = 0$.
* Ví dụ: Phương trình hồi quy tuyến tính có dạng $\hat y=\mathbf{X}\theta$.

### Công thức tách dòng (Block)

Khi có một phương trình phức tạp hoặc quan trọng vàmuốn nó nằm riêng một dòng và căn giữa, dùng hai cặp dấu đô-la `$$`.

Ví dụ về phương trình hàm MSE:

$$
\frac{1}{n}\sum_{i=1}^{n} (\hat y - y)^2
$$

Ví dụ hàm BCE (Binary Cross-Enotrpy):

$$
-y\log{\hat y} - (1-y)\log{(1-\hat y)}
$$

## 4. Phần code

Dùng dấu huyền (backtick) `như thế này`.

Còn nếu là một khối code (code block), dùng ba dấu huyền ``` và ghi rõ ngôn ngữ (như `python` hay `js`) để nó format:

```python
def say_hello(name):
    # Đây là một comment trong code
    print(f"Hello, {name}!")

say_hello("World")
```

```bash
!pip install requirement.txt
```