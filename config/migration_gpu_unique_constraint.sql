-- ============================================
-- 迁移脚本：为 gpu_allocations 表添加 DB 层并发互斥约束
-- ============================================
-- 背景：原表的 gpu_allocate() 是无条件 INSERT，没有任何约束阻止两个并发请求
--      同时拿到同一块 GPU 的 'allocated' 记录（仅在应用层展示为列表，不构成真正互斥）。
-- 适用对象：已存在的数据库（新建数据库请直接使用 database_init_with_auth.sql，已包含此约束）。
-- 用法：mysql -u root -p flaskdemo_user < config/migration_gpu_unique_constraint.sql

USE flaskdemo_user;

-- 执行前自查：如果同一 gpu_id 已存在多条 'allocated' 记录（历史脏数据），
-- 添加唯一索引会失败，需先手动核实保留一条、其余更新为 'released'。
-- SELECT gpu_id, COUNT(*) FROM gpu_allocations WHERE status='allocated' GROUP BY gpu_id HAVING COUNT(*) > 1;

ALTER TABLE gpu_allocations
    ADD COLUMN active_gpu_id INT GENERATED ALWAYS AS (CASE WHEN status = 'allocated' THEN gpu_id ELSE NULL END) STORED
        COMMENT '仅 allocated 状态时等于 gpu_id，否则为 NULL；配合唯一索引在 DB 层防止同一 GPU 被并发重复占用',
    ADD UNIQUE KEY uniq_active_gpu (active_gpu_id);
