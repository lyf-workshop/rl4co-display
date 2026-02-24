#!/usr/bin/env python3
"""
数据库清除脚本
用于清除 rl4co-display 项目数据库中的数据

用法：
    python scripts/clear_database.py              # 交互式菜单
    python scripts/clear_database.py --all        # 清除所有数据（保留表结构）
    python scripts/clear_database.py --training   # 只清除训练数据（保留用户）
    python scripts/clear_database.py --user 3     # 清除指定用户的数据
    python scripts/clear_database.py --reset      # 重建数据库（删除+重建所有表）
"""

import os
import sys
import shutil
import argparse
import mysql.connector

# ── 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ── 数据库配置（优先从 config.py 读取，其次环境变量）
try:
    from config.config import Config as _AppConfig
    _host     = _AppConfig.MYSQL_HOST
    _user     = _AppConfig.MYSQL_USER
    _password = _AppConfig.MYSQL_PASSWORD
    _db       = _AppConfig.MYSQL_DB
except Exception:
    _host     = 'localhost'
    _user     = 'root'
    _password = ''
    _db       = 'flaskdemo_user'

DB_CONFIG = {
    'host':     os.getenv('MYSQL_HOST',     _host),
    'user':     os.getenv('MYSQL_USER',     _user),
    'password': os.getenv('MYSQL_PASSWORD', _password),
    'database': os.getenv('MYSQL_DB',       _db),
    'autocommit': False,
}

# ── 用户文件目录
PLOTS_BASE = os.path.join(PROJECT_ROOT, 'static', 'model_plots')
CHECKPOINTS_BASE = os.path.join(PROJECT_ROOT, 'checkpoints')


# ──────────────────────────────────────────
# 颜色输出
# ──────────────────────────────────────────
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

def ok(msg):   print(f'{GREEN}✅ {msg}{RESET}')
def warn(msg): print(f'{YELLOW}⚠️  {msg}{RESET}')
def err(msg):  print(f'{RED}❌ {msg}{RESET}')
def info(msg): print(f'{CYAN}ℹ️  {msg}{RESET}')
def title(msg):print(f'\n{BOLD}{msg}{RESET}')


# ──────────────────────────────────────────
# 数据库连接
# ──────────────────────────────────────────
def get_conn():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as e:
        err(f'数据库连接失败: {e}')
        print('\n请检查：')
        print('  1. MySQL 服务是否已启动')
        print('  2. 密码是否正确（当前: config.py 中的 MYSQL_PASSWORD）')
        print('  3. 数据库 flaskdemo_user 是否存在')
        sys.exit(1)


# ──────────────────────────────────────────
# 统计信息
# ──────────────────────────────────────────
def show_stats(conn):
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT COUNT(*) as n FROM users")
    users = cursor.fetchone()['n']
    cursor.execute("SELECT COUNT(*) as n FROM training_sessions")
    sessions = cursor.fetchone()['n']
    cursor.execute("SELECT COUNT(*) as n FROM training_files")
    files = cursor.fetchone()['n']
    cursor.execute("SELECT COALESCE(SUM(file_size),0) as s FROM training_files")
    size = cursor.fetchone()['s']
    cursor.close()

    print(f'''
  当前数据库状态：
    用户数量      : {BOLD}{users}{RESET} 个
    训练会话      : {BOLD}{sessions}{RESET} 条
    文件记录      : {BOLD}{files}{RESET} 条
    记录总大小    : {BOLD}{size / 1024 / 1024:.2f}{RESET} MB
''')


def show_users(conn):
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT u.id, u.username, u.email, u.create_time,
               COUNT(DISTINCT ts.id) as sessions,
               COUNT(tf.id) as files
        FROM users u
        LEFT JOIN training_sessions ts ON u.id = ts.user_id
        LEFT JOIN training_files tf ON u.id = tf.user_id
        GROUP BY u.id
        ORDER BY u.id
    """)
    rows = cursor.fetchall()
    cursor.close()

    if not rows:
        info('暂无用户')
        return

    print(f'  {"ID":<5} {"用户名":<15} {"会话":<6} {"文件":<6} {"注册时间"}')
    print('  ' + '-' * 55)
    for r in rows:
        ct = r['create_time'].strftime('%Y-%m-%d %H:%M') if r['create_time'] else '-'
        print(f'  {r["id"]:<5} {r["username"]:<15} {r["sessions"]:<6} {r["files"]:<6} {ct}')


# ──────────────────────────────────────────
# 删除磁盘文件
# ──────────────────────────────────────────
def delete_user_files(user_id):
    """删除某用户的所有磁盘文件（图片/GIF/checkpoint）"""
    count = 0

    plots_dir = os.path.join(PLOTS_BASE, f'user_{user_id}')
    if os.path.exists(plots_dir):
        try:
            shutil.rmtree(plots_dir)
            count += 1
            ok(f'  已删除 {plots_dir}')
        except Exception as e:
            warn(f'  删除 {plots_dir} 失败: {e}')

    ckpt_dir = os.path.join(CHECKPOINTS_BASE, f'user_{user_id}')
    if os.path.exists(ckpt_dir):
        try:
            shutil.rmtree(ckpt_dir)
            count += 1
            ok(f'  已删除 {ckpt_dir}')
        except Exception as e:
            warn(f'  删除 {ckpt_dir} 失败: {e}')

    return count


def delete_all_files():
    """删除所有用户的磁盘文件"""
    count = 0
    for base in [PLOTS_BASE, CHECKPOINTS_BASE]:
        if not os.path.exists(base):
            continue
        for entry in os.listdir(base):
            if entry.startswith('user_'):
                path = os.path.join(base, entry)
                try:
                    shutil.rmtree(path)
                    count += 1
                    ok(f'  已删除 {path}')
                except Exception as e:
                    warn(f'  删除 {path} 失败: {e}')
    return count


# ──────────────────────────────────────────
# 数据库清除操作
# ──────────────────────────────────────────
def clear_training_data(conn, delete_files=True):
    """清除所有训练数据（training_sessions + training_files），保留用户"""
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM training_files")  # noqa: S608
        files_deleted = cursor.rowcount
        cursor.execute("DELETE FROM training_sessions")  # noqa: S608
        sessions_deleted = cursor.rowcount
        conn.commit()
        ok(f'已删除 {sessions_deleted} 条训练会话记录')
        ok(f'已删除 {files_deleted} 条文件记录')

        if delete_files:
            title('删除磁盘文件...')
            delete_all_files()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def clear_all_data(conn, delete_files=True):
    """清除所有数据，包括用户（表结构保留）"""
    cursor = conn.cursor()
    try:
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        cursor.execute("DELETE FROM training_files")  # noqa: S608
        f = cursor.rowcount
        cursor.execute("DELETE FROM training_sessions")  # noqa: S608
        s = cursor.rowcount
        cursor.execute("DELETE FROM users")  # noqa: S608
        u = cursor.rowcount
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        conn.commit()
        ok(f'已删除 {u} 个用户')
        ok(f'已删除 {s} 条训练会话记录')
        ok(f'已删除 {f} 条文件记录')

        if delete_files:
            title('删除磁盘文件...')
            delete_all_files()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def clear_user_data(conn, user_id, delete_files=True):
    """清除指定用户的所有数据"""
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, username FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            err(f'用户 ID={user_id} 不存在')
            return False

        cursor.execute("DELETE FROM training_files WHERE user_id = %s", (user_id,))
        f = cursor.rowcount
        cursor.execute("DELETE FROM training_sessions WHERE user_id = %s", (user_id,))
        s = cursor.rowcount
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()

        ok(f'已删除用户 {user["username"]} (ID={user_id})')
        ok(f'已删除 {s} 条训练会话')
        ok(f'已删除 {f} 条文件记录')

        if delete_files:
            title(f'删除用户 {user_id} 的磁盘文件...')
            delete_user_files(user_id)
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def reset_database(conn):
    """完全重建数据库（删除所有表后重新创建）"""
    cursor = conn.cursor()
    sql_path = os.path.join(PROJECT_ROOT, 'config', 'database_init_with_auth.sql')

    try:
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        for table in ['training_files', 'training_sessions', 'users']:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            ok(f'已删除表 {table}')
        for view in ['user_training_stats']:
            cursor.execute(f"DROP VIEW IF EXISTS {view}")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        conn.commit()

        ok('旧表已全部删除，正在重建...')

        # 执行初始化 SQL
        if os.path.exists(sql_path):
            with open(sql_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()

            # 逐条执行（跳过注释和空行，处理分隔符）
            statements = []
            current = []
            for line in sql_content.splitlines():
                stripped = line.strip()
                if stripped.startswith('--') or stripped.startswith('#') or not stripped:
                    continue
                if stripped.upper() in ('DELIMITER //', 'DELIMITER ;'):
                    continue
                current.append(line)
                if stripped.endswith(';') or stripped.endswith('//'):
                    stmt = '\n'.join(current).strip().rstrip(';').rstrip('//')
                    if stmt:
                        statements.append(stmt)
                    current = []

            executed = 0
            for stmt in statements:
                try:
                    cursor.execute(stmt)
                    conn.commit()
                    executed += 1
                except mysql.connector.Error as e:
                    if e.errno not in (1050, 1060, 1061):  # 忽略"已存在"类错误
                        warn(f'SQL 执行警告: {e}')

            ok(f'已执行 {executed} 条 SQL 语句')
        else:
            warn(f'未找到初始化 SQL: {sql_path}，将手动重建基础表')
            _create_tables_manually(cursor, conn)

        # 修复 ENUM 包含 comparison 类型
        _fix_enum(cursor, conn)

    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()

    # 删除所有磁盘文件
    title('删除所有磁盘文件...')
    delete_all_files()


def _create_tables_manually(cursor, conn):
    """手动建表（backup）"""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            email VARCHAR(100),
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_sessions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(50) UNIQUE NOT NULL,
            user_id INT NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            problem_type VARCHAR(50) NOT NULL,
            config JSON,
            status ENUM('running','completed','failed') DEFAULT 'running',
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP NULL,
            final_loss DECIMAL(10,4),
            final_reward DECIMAL(10,4),
            best_reward DECIMAL(10,4),
            checkpoint_path VARCHAR(255),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_files (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            session_id VARCHAR(50) NOT NULL,
            file_name VARCHAR(255) NOT NULL,
            file_type ENUM('plot','comparison','animation','curve','checkpoint') NOT NULL,
            file_path VARCHAR(500) NOT NULL,
            file_size BIGINT,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    conn.commit()
    ok('基础表创建完成')


def _fix_enum(cursor, conn):
    """修复 training_files.file_type ENUM，确保包含 comparison 类型"""
    try:
        cursor.execute("""
            SELECT COLUMN_TYPE FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = 'training_files'
              AND COLUMN_NAME = 'file_type'
        """, (DB_CONFIG['database'],))
        row = cursor.fetchone()
        if row and 'comparison' not in row[0]:
            cursor.execute("""
                ALTER TABLE training_files
                MODIFY COLUMN file_type
                ENUM('plot','comparison','animation','curve','checkpoint') NOT NULL
            """)
            conn.commit()
            ok("已修复 training_files.file_type ENUM（添加 'comparison'）")
    except Exception as e:
        warn(f'修复 ENUM 失败（非致命）: {e}')


# ──────────────────────────────────────────
# 交互式菜单
# ──────────────────────────────────────────
def interactive_menu():
    conn = get_conn()

    while True:
        title('=' * 50)
        print(f'{BOLD}  RL4CO Display 数据库清除工具{RESET}')
        title('=' * 50)
        show_stats(conn)

        print('  操作选项：')
        print('  [1] 查看所有用户')
        print('  [2] 清除训练数据（保留用户账号）')
        print('  [3] 清除所有数据（包括用户）')
        print('  [4] 清除指定用户的数据')
        print('  [5] 修复数据库表结构（ENUM等）')
        print('  [6] 完全重建数据库')
        print('  [0] 退出')
        print()

        choice = input('  请选择操作 [0-6]: ').strip()

        if choice == '0':
            print()
            ok('已退出')
            break

        elif choice == '1':
            title('所有用户列表')
            show_users(conn)

        elif choice == '2':
            title('清除训练数据（保留用户）')
            warn('将删除所有 training_sessions、training_files 记录以及磁盘上的图片/GIF/checkpoint 文件')
            confirm = input('  确认操作？输入 yes 继续：').strip().lower()
            if confirm == 'yes':
                clear_training_data(conn)
                ok('训练数据清除完成')
            else:
                info('已取消')

        elif choice == '3':
            title('清除所有数据（包括用户）')
            warn('将删除 users、training_sessions、training_files 所有记录，以及磁盘文件')
            confirm = input('  确认操作？输入 yes 继续：').strip().lower()
            if confirm == 'yes':
                clear_all_data(conn)
                ok('所有数据清除完成')
            else:
                info('已取消')

        elif choice == '4':
            title('清除指定用户数据')
            show_users(conn)
            uid = input('\n  请输入要删除的用户 ID：').strip()
            if uid.isdigit():
                confirm = input(f'  确认删除用户 ID={uid} 的所有数据？输入 yes 继续：').strip().lower()
                if confirm == 'yes':
                    clear_user_data(conn, int(uid))
                else:
                    info('已取消')
            else:
                err('请输入有效的数字 ID')

        elif choice == '5':
            title('修复数据库表结构')
            cursor = conn.cursor()
            _fix_enum(cursor, conn)
            cursor.close()

        elif choice == '6':
            title('完全重建数据库')
            warn('将删除所有表并重新初始化，所有数据和文件将被清除！')
            confirm = input('  确认操作？输入 RESET 继续：').strip()
            if confirm == 'RESET':
                reset_database(conn)
                ok('数据库重建完成')
            else:
                info('已取消')

        else:
            err('无效选项，请重新选择')

    conn.close()


# ──────────────────────────────────────────
# 命令行参数入口
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='RL4CO Display 数据库清除工具')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all',      action='store_true', help='清除所有数据（包含用户，保留表结构）')
    group.add_argument('--training', action='store_true', help='只清除训练数据（保留用户账号）')
    group.add_argument('--user',     type=int, metavar='USER_ID', help='清除指定用户的数据')
    group.add_argument('--fix',      action='store_true', help='只修复表结构（ENUM等）')
    group.add_argument('--reset',    action='store_true', help='完全重建数据库（删除+重建所有表）')
    parser.add_argument('--no-files', action='store_true', help='跳过磁盘文件删除')
    args = parser.parse_args()

    # 无参数 → 交互菜单
    if not any([args.all, args.training, args.user, args.fix, args.reset]):
        interactive_menu()
        return

    conn = get_conn()
    delete_files = not args.no_files

    try:
        if args.training:
            title('清除训练数据（保留用户）')
            clear_training_data(conn, delete_files)
            ok('完成')

        elif args.all:
            title('清除所有数据')
            clear_all_data(conn, delete_files)
            ok('完成')

        elif args.user:
            title(f'清除用户 ID={args.user} 的数据')
            clear_user_data(conn, args.user, delete_files)
            ok('完成')

        elif args.fix:
            title('修复表结构')
            cursor = conn.cursor()
            _fix_enum(cursor, conn)
            cursor.close()
            ok('完成')

        elif args.reset:
            title('重建数据库')
            reset_database(conn)
            ok('完成')

    except Exception as e:
        err(f'操作失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
