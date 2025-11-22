
# I. Vấn đề của bài toán

Trong các bài toán phân loại nhị phân kinh điển, chẳng hạn như phân biệt *spam / không spam* hay *bệnh / không bệnh*, mô hình **logistic** tỏ ra đặc biệt hiệu quả. Hàm *sigmoid* nhận một giá trị logit và biến nó thành xác suất thuộc lớp “1”, trong khi xác suất thuộc lớp “0” chỉ đơn giản là phần bù còn lại. Cách biểu diễn này vừa trực quan, vừa phù hợp với trực giác xác suất, nên logistic nhanh chóng trở thành lựa chọn mặc định cho rất nhiều bài toán thực tế khi chỉ có hai lớp cần phân biệt.

![Phân loại đa lớp](image/image_1.png)

Tuy nhiên, khi chuyển sang các bài toán phân loại nhiều lớp – chẳng hạn mô hình cần phân biệt giữa *mèo, chó, chim, cá* – việc “cố gắng” sử dụng lại logistic theo cách truyền thống bắt đầu bộc lộ nhiều hạn chế. Một phương pháp thường gặp là huấn luyện nhiều bộ phân loại nhị phân theo chiến lược *one-vs-rest*, trong đó mỗi mô hình đảm nhận nhiệm vụ dự đoán “mèo hay không mèo”, “chó hay không chó”,… Từ đó, mỗi lớp tạo ra một giá trị xác suất riêng biệt. Vấn đề nằm ở chỗ các xác suất này **không có bất kỳ ràng buộc nào với nhau**: tổng có thể lớn hơn 1, nhiều lớp có thể cùng nhận giá trị rất cao, và mô hình không thật sự bị buộc phải lựa chọn một lớp nổi trội nhất. Điều này khiến việc diễn giải kết quả trở nên thiếu nhất quán: một giá trị xác suất cao không còn đại diện rõ ràng cho khả năng “lớp này tốt nhất” khi đặt trong bối cảnh cạnh tranh giữa toàn bộ các lớp. Kết quả cuối cùng dễ rơi vào tình trạng mơ hồ và thiếu tính xác suất với bài toán đa lớp.

Chính từ những vấn đề đó thì có một cơ chế vừa tổng quát hóa được logistic, vừa ép toàn bộ dự đoán nằm trên một phân phối xác suất với sự cạnh tranh rõ ràng giữa các lớp, hàm softmax được ra đời. Softmax cho phép biến toàn bộ vector logit thành một phân phối xác suất chuẩn hóa trên K lớp, qua đó khắc phục các nhược điểm của các mô hình logistic trước đây và đặt nền tảng cho các kiến trúc phân loại đa lớp hiện đại trong học máy và deep learning.
