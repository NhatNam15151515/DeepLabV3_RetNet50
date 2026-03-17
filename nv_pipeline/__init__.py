"""NutritionVerse-style multi-tier pipeline (segmentation → depth/volume → weight).

Các module chính:
- tier1_segmentation.py   : trích xuất instance từ semantic mask
- tier2_depth_volume.py   : ước lượng depth + volume bằng MiDaS
- tier3_weight_estimation : ước lượng trọng lượng từ thể tích
- pipeline.py             : ghép 3 tầng thành end-to-end pipeline

Hiện tại đây là code tham khảo/experimental. Khi tích hợp lại với hệ thống chính,
hãy kiểm tra lại config, class map và đường dẫn model trước khi dùng trong production.
"""

