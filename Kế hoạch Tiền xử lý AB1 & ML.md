# **KẾ HOẠCH CHI TIẾT: TIỀN XỬ LÝ AB1 TRACE & XÂY DỰNG DATASET ML-READY (PHASE 1\)**

## **1\. Tổng quan Dự án (Project Overview)**

**Bối cảnh:** Các công cụ xử lý Sanger sequencing (AB1 trace) hiện tại như Tracy, sangeranalyseR chủ yếu tập trung cắt bỏ (trimming) vùng nhiễu ở hai đầu chuỗi. Tuy nhiên, đối với mẫu DNA bị phân hủy (degraded DNA, hài cốt), hiện tượng nhiễu thường xuyên xuất hiện ở **giữa trace** (middle-trace noise: dye blobs, baseline drift, low SNR). Điều này dẫn đến tỷ lệ false positive/negative cao và gọi sai heteroplasmy.

**Mục tiêu kép (Dual-purpose) của Phase 1:**

1. **Giải quyết tức thì (Deterministic):** Xây dựng pipeline tiền xử lý và gắn cờ (flagging) tự động các vùng nhiễu (noisy regions) một cách có quy tắc, minh bạch và có thể kiểm toán. Mục tiêu giảm ≥40% false positive/negative trên tập degraded batch.  
2. **Chuẩn bị cho Machine Learning (ML-Ready):** Trích xuất các đặc trưng (features) phong phú và tạo nhãn giả (pseudo-labels) để xây dựng tập dữ liệu huấn luyện, chuẩn bị cho việc áp dụng ML/AI ở Phase 2 nhằm nhận diện nhiễu tinh vi hơn.

## **2\. Bản chất Bài toán & Hướng tiếp cận**

* **Input:** Raw AB1 trace (4 kênh intensity A/C/G/T theo scan point) \+ Reference rCRS \+ Primer info.  
* **Rào cản Groundtruth:** Hiện tại dự án chỉ có kết quả variant cuối cùng (từ manual review/Sequencher), không có nhãn nhiễu (noisy labels) ở cấp độ từng base/region.  
* **Giải pháp:** Phase 1 sẽ tạo **Pseudo-labels** (dựa trên rule-based heuristic) và **Feature Matrix**. Sau này, ML model sẽ dùng variant groundtruth làm "weak supervision" (proxy: vùng có variant gọi sai thường là vùng nhiễu).  
* **Chiến lược xử lý cốt lõi:** Thay vì xử lý Global, pipeline bắt buộc phải xử lý và tính toán Quality Control (QC) metrics theo **Per-window (50-100 bp)**.

## **3\. Thiết kế Dữ liệu: Input & Output (ML-Ready Data Specification)**

Để đảm bảo phục vụ cả pipeline hiện tại và ML model tương lai, đầu ra của Phase 1 được quy định như sau:

### **3.1. Deterministic Output (Dùng ngay cho Pipeline hiện hành)**

* **Clean Trace:** Dữ liệu sóng đã qua hiệu chỉnh đường nền (baseline), làm mượt và chuẩn hóa.  
* **Noisy Mask:** Array boolean đánh dấu các vùng nhiễu.  
* **Variant Flags:** Các variant rơi vào vùng nhiễu sẽ được gán is\_noisy=True, qc\_pass=False và cảnh báo level 4\.  
* **Báo cáo:** Cập nhật file statistic\_fullbatch.json và Comparator report với cột "Noisy Flag".

### **3.2. ML-Ready Dataset (Dùng cho Phase 2\)**

Export dưới định dạng JSON/Parquet chứa cấu trúc sau:

{  
  "sample\_id": "Mẫu\_hài\_cốt\_001",  
  "per\_base": \[   
    { "pos": 16189, "SNR": 12.3, "messiness": 2.1, "noisy\_pseudo\_label": 1, "variant\_groundtruth": "C\>T" }   
  \],  
  "per\_window": \[   
    { "start": 300, "end": 350, "SNR\_avg": 8.5, "noisy": true }  
  \],  
  "noisy\_regions": \[\[305, 320\], \[16180, 16193\]\]  
}

## **4\. Kế hoạch Triển khai Chi tiết (3 Sub-phases)**

### **Sub-phase 1.1: Phân tích Dữ liệu Thực tế & Feature Design**

* **Thời gian:** 1 – 2 ngày.  
* **Nhiệm vụ:**  
  * Trích xuất tập mẫu test: 10–20 AB1 degraded samples \+ Positive Control.  
  * Tính toán bộ QC metrics trên từng **window 50-bp** (quét toàn bộ trace).  
  * Xác định các pattern nhiễu điển hình (Dye blobs 60–140 bp, middle drift, polyC stutter).  
  * Chốt danh sách **10–12 features cuối cùng** (per-base & per-window) bao gồm: Local SNR, baseline variance, peak ratio (primary/secondary), messiness index, dye-blob probability, distance to nearest polyC...  
* **Deliverable:** Báo cáo phân tích \+ Biểu đồ minh họa (Raw vs Noisy positions) \+ Danh sách Features chốt hạ.

### **Sub-phase 1.2: Thiết kế Preprocessing Pipeline & Feature Extraction**

* **Thời gian:** 2 – 3 ngày.  
* **Nhiệm vụ:**  
  * Xây dựng module tiền xử lý (Dự kiến tại src/utils/seq.py).  
  * Cài đặt thứ tự pipeline bắt buộc:  
    1. **Baseline correction:** Hiệu chỉnh đường nền toàn trace (ALS hoặc polynomial).  
    2. **Noise smoothing:** Làm mượt nhiễu bằng thuật toán Savitzky-Golay.  
    3. **Per-channel normalization:** Chuẩn hóa theo từng kênh màu.  
    4. **Dye blob mask:** Nhận diện và mask vùng nhiễu thuốc nhuộm (thường ở 60–140 bp).  
  * *Đồng thời:* Trích xuất Full Feature Vector trong quá trình chạy.  
* **Deliverable:** Module Python hoàn chỉnh \+ Output gồm Clean trace, Noisy\_mask, Feature matrix (JSON/Parquet) test pass.

### **Sub-phase 1.3: Regional Noisy Flagging, Pseudo-labeling & Integration**

* **Thời gian:** 3 – 4 ngày.  
* **Nhiệm vụ:**  
  * Merge các window nhiễu thành noisy\_regions cụ thể.  
  * **Tích hợp vào Tracy:** Sửa hàm post\_process(). Nếu variant nằm trong noisy\_regions \-\> đánh cờ is\_noisy=True và qc\_pass=False.  
  * **Tạo Pseudo-labels cho ML:** Sinh nhãn (0/1) dựa trên rule-based heuristic.  
  * **Cập nhật hệ thống:** Bổ sung vào model SampleAnalysisResult, file statistic\_fullbatch.json và Comparator report.  
* **Deliverable:** Toàn bộ pipeline tích hợp thành công \+ Dataset ML đầu tiên sẵn sàng.

## **5\. Timeline & Phân công Công việc**

| Tuần | Sub-phase | Task chính | Người phụ trách (Owner) | Đầu ra (Deliverable) |
| :---- | :---- | :---- | :---- | :---- |
| **Tuần 1** | 1.1 | Phân tích signal \+ Design features | P1 (T1-A Owner) | Báo cáo features \+ Biểu đồ |
| **Tuần 1-2** | 1.2 | Implement preprocessing \+ Feature extraction | P2 | Module code \+ Pass test local |
| **Tuần 2** | 1.3 | Flagging \+ Pseudo-labels \+ Integration | P3 | Full pipeline \+ ML Dataset (v1) |
| **Tuần 2** | *Benchmark* | Đánh giá Before/After trên toàn batch | Toàn team | Báo cáo giảm ≥40% False Positive |

## **6\. Definition of Done (Tiêu chuẩn Hoàn thành)**

1. Lệnh make test pass toàn bộ (bao gồm các test case mới cho noisy regions và tính nhất quán của features).  
2. Code tuân thủ chuẩn Ruff clean.  
3. Báo cáo Benchmark chứng minh hiệu quả giảm thiểu false variant trên Degraded Batch.  
4. Cơ chế Noisy Flag hoạt động chủ động (Proactive): Từ chối gọi variant hoặc cảnh báo chính xác trên vùng nhiễu giữa trace.  
5. ML Dataset version 1.0 (JSON/Parquet) sẵn sàng, bao gồm: Features \+ Pseudo-labels \+ Groundtruth variants.

## **7\. Quản trị Rủi ro (Risks & Mitigations)**

* **Rủi ro 1: Ngưỡng (Threshold) quá gắt làm mất variant thật.**  
  * *Khắc phục:* Thiết kế hệ thống threshold có thể tùy chỉnh (adjustable). Sử dụng Positive Control validation để cân chỉnh.  
* **Rủi ro 2: Preprocessing làm chậm tốc độ xử lý Batch.**  
  * *Khắc phục:* Đảm bảo module preprocessing chỉ chạy đúng 1 lần trên mỗi file AB1 và tối ưu hóa cho xử lý song song (multiprocessing).