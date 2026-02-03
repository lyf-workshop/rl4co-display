"""
测试数据集解析功能
"""
import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app_files import parse_dataset


class TestParseDataset:
    """测试parse_dataset函数"""
    
    def test_parse_json_format(self):
        """测试JSON格式解析"""
        json_content = '{"coordinates": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}'
        result = parse_dataset(json_content, 'json')
        assert result is not None
        assert len(result) == 3
        assert result[0] == [1.0, 2.0]
        assert result[2] == [5.0, 6.0]
    
    def test_parse_txt_format(self):
        """测试TXT格式解析"""
        txt_content = """1.0 2.0
3.0 4.0
5.0 6.0"""
        result = parse_dataset(txt_content, 'txt')
        assert result is not None
        assert len(result) == 3
        assert result[0] == [1.0, 2.0]
    
    def test_parse_txt_with_comments(self):
        """测试TXT格式（带注释）解析"""
        txt_content = """# This is a comment
1.0 2.0
# Another comment
3.0 4.0"""
        result = parse_dataset(txt_content, 'txt')
        assert result is not None
        assert len(result) == 2
    
    def test_parse_tsp_format(self):
        """测试TSPLIB格式解析"""
        tsp_content = """NAME : test
TYPE : TSP
DIMENSION : 3
NODE_COORD_SECTION
1 10.0 20.0
2 30.0 40.0
3 50.0 60.0
EOF"""
        result = parse_dataset(tsp_content, 'tsp')
        assert result is not None
        assert len(result) == 3
        assert result[0] == [10.0, 20.0]
        assert result[2] == [50.0, 60.0]
    
    def test_parse_invalid_json(self):
        """测试无效JSON"""
        invalid_json = '{"invalid": json}'
        result = parse_dataset(invalid_json, 'json')
        assert result is None
    
    def test_parse_empty_content(self):
        """测试空内容"""
        result = parse_dataset('', 'txt')
        assert result is None
    
    def test_parse_json_missing_coordinates(self):
        """测试JSON缺少coordinates字段"""
        json_content = '{"data": [[1.0, 2.0]]}'
        result = parse_dataset(json_content, 'json')
        assert result is None


