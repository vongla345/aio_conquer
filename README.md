# Repo blog bài học

Đây là nơi lưu trữ các bài blog ngắn về những chủ đề đang tìm hiểu trong AIO.

## 🚀 Hướng dẫn sử dụng Git

### 1. Tải (Clone) kho về máy

Thực hiện **một lần duy nhất** để lấy toàn bộ nội dung.

1.  **Cài Git:** Tải từ [git-scm.com](https://git-scm.com/) nếu chưa có.
2.  **Mở Terminal:** Mở `Terminal` hoặc `Git Bash`.
3.  **Chạy lệnh:** (Thay `[LINK_HTTPS_CUA_REPO]` bằng link đã sao chép)

    ```bash
    git clone [LINK_HTTPS_CUA_REPO]
    ```

### 2. Cập nhật (Pull) nội dung mới

Dùng khi repo này có cập nhật mới và muốn đồng bộ về máy.

1.  Mở Terminal, đi vào thư mục dự án (dùng `cd TenThuMucDuAn`).
2.  Chạy lệnh `pull`:

    ```bash
    git pull origin main
    ```

    *(Lưu ý: kiểm tra tên nhánh trước khi pull)*

---

## 🤝 Hướng dẫn đóng góp (Cho thành viên)

Đây là quy trình chuẩn để cập nhật nội dung khi có quyền truy cập (write access) vào repo này. Vui lòng **không** làm việc trực tiếp trên nhánh `main`.

Trước khi cập nhật, đảm bảo đã clone repo này trước đó

### Quy trình làm việc (Branch -> PR)

#### Bước 1: Đồng bộ nhánh `main`

Trước khi bắt đầu, luôn đảm bảo nhánh `main` trên máy (local) là mới nhất.

```bash
git checkout main
git pull origin main
```

#### Bước 2: Tạo nhánh (Branch) mới

Tạo một nhánh mới từ main cho tính năng hoặc bài viết của bạn. Đặt tên nhánh rõ ràng (ví dụ: them-bai-hoc-git hoặc sua-loi-bai-A).

```bash
# Ví dụ: git checkout -b them-bai-hoc-git
git checkout -b ten-nhanh-moi
```

#### Bước 3: Chỉnh sửa và Lưu (Commit)**

Thực hiện các thay đổi (thêm/sửa file). Sau đó, lưu lại các thay đổi đó (commit).

```bash
# Thêm tất cả các file đã thay đổi
git add .

# Ghi lại thay đổi với một lời nhắn
git commit -m "Nội dung mô tả thay đổi (ví dụ: Thêm bài học Git)"
```

#### Bước 4: Push nhánh lên repo chung

Đẩy nhánh mới lên repo chung trên GitHub (các thay đổi và commit trước đó chỉ cập nhật trên máy local của cá nhân thôi).

```bash
git push origin ten-nhanh-moi
```

#### Bước 5: Tạo Pull Request (PR)

Mở repo này trên GitHub, sẽ thấy một thanh thông báo màu vàng với nút "Compare & pull request". 

(Nếu không thấy) Mở tab "Pull Requests" -> nhấn "New pull request".

Chọn nhánh vừa push (ten-nhanh-moi) so sánh với nhánh main.

Tạo PR, viết mô tả rõ ràng.

#### Bước 6: Merge và Xóa nhánh

Sau khi PR được duyệt (approved) và không có xung đột (conflict), hãy nhấn "Merge pull request" để gộp vào main.

Sau khi gộp, có thể an toàn xóa nhánh đã làm việc trên GitHub và trên máy.
