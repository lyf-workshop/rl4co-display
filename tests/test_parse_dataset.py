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
    """测试parse_dataset函数（TSP，向后兼容）"""

    def test_parse_json_format(self):
        """测试JSON格式解析"""
        content = '{"coordinates": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}'
        result = parse_dataset(content, 'json', 'tsp')
        assert result is not None
        assert len(result['coordinates']) == 3
        assert result['coordinates'][0] == [1.0, 2.0]
        assert result['coordinates'][2] == [5.0, 6.0]

    def test_parse_txt_format(self):
        """测试TXT格式解析"""
        result = parse_dataset("1.0 2.0\n3.0 4.0\n5.0 6.0", 'txt', 'tsp')
        assert result is not None
        assert len(result['coordinates']) == 3
        assert result['coordinates'][0] == [1.0, 2.0]

    def test_parse_txt_with_comments(self):
        """测试TXT格式（带注释）解析"""
        txt_content = "# This is a comment\n1.0 2.0\n# Another comment\n3.0 4.0"
        result = parse_dataset(txt_content, 'txt', 'tsp')
        assert result is not None
        assert len(result['coordinates']) == 2

    def test_parse_tsp_format(self):
        """测试TSPLIB格式解析"""
        tsp_content = "NAME : test\nNODE_COORD_SECTION\n1 10.0 20.0\n2 30.0 40.0\n3 50.0 60.0\nEOF"
        result = parse_dataset(tsp_content, 'tsp', 'tsp')
        assert result is not None
        assert len(result['coordinates']) == 3
        assert result['coordinates'][0] == [10.0, 20.0]
        assert result['coordinates'][2] == [50.0, 60.0]

    def test_parse_invalid_json(self):
        """测试无效JSON"""
        result = parse_dataset('{"invalid": json}', 'json', 'tsp')
        assert result is None

    def test_parse_empty_content(self):
        """测试空内容"""
        result = parse_dataset('', 'txt', 'tsp')
        assert result is None

    def test_parse_json_missing_coordinates(self):
        """测试JSON缺少coordinates字段"""
        result = parse_dataset('{"data": [[1.0, 2.0]]}', 'json', 'tsp')
        assert result is None


class TestParseDatasetMultiProblem:
    """测试多问题类型的数据集解析"""

    def test_tsp_returns_dict(self):
        """TSP 解析结果是 dict"""
        content = '{"coordinates": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}'
        result = parse_dataset(content, 'json', 'tsp')
        assert isinstance(result, dict)
        assert result['coordinates'] == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    def test_cvrp_with_demands(self):
        """CVRP 解析含 demands 字段"""
        content = '{"coordinates": [[0.1, 0.2], [0.3, 0.4]], "demands": [0.3, 0.5]}'
        result = parse_dataset(content, 'json', 'cvrp')
        assert result is not None
        assert result['demands'] == [0.3, 0.5]
        assert result['coordinates'] == [[0.1, 0.2], [0.3, 0.4]]

    def test_cvrp_demands_out_of_range(self):
        """CVRP demands 超出 (0,1] 范围时返回 None"""
        content = '{"coordinates": [[0.1, 0.2], [0.3, 0.4]], "demands": [1.5, 0.5]}'
        result = parse_dataset(content, 'json', 'cvrp')
        assert result is None

    def test_pdp_odd_coordinates_fails(self):
        """PDP 奇数个坐标返回 None"""
        content = '{"coordinates": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}'
        result = parse_dataset(content, 'json', 'pdp')
        assert result is None

    def test_pdp_even_coordinates_ok(self):
        """PDP 偶数个坐标通过"""
        content = '{"coordinates": [[0.1, 0.2], [0.3, 0.4]]}'
        result = parse_dataset(content, 'json', 'pdp')
        assert result is not None
        assert len(result['coordinates']) == 2

    def test_vrptw_with_time_windows(self):
        """VRPTW 解析含 time_windows"""
        content = '{"coordinates": [[0.1, 0.2]], "time_windows": [[0.0, 0.5]]}'
        result = parse_dataset(content, 'json', 'vrptw')
        assert result is not None
        assert result['time_windows'] == [[0.0, 0.5]]

    def test_vrptw_invalid_time_window(self):
        """VRPTW time_window start >= end 时返回 None"""
        content = '{"coordinates": [[0.1, 0.2]], "time_windows": [[0.8, 0.3]]}'
        result = parse_dataset(content, 'json', 'vrptw')
        assert result is None

    def test_op_with_prizes(self):
        """OP 解析含 prizes"""
        content = '{"coordinates": [[0.1, 0.2], [0.3, 0.4]], "prizes": [1.0, 2.0]}'
        result = parse_dataset(content, 'json', 'op')
        assert result is not None
        assert result['prizes'] == [1.0, 2.0]

    def test_op_negative_prize_fails(self):
        """OP prizes 包含负值时返回 None"""
        content = '{"coordinates": [[0.1, 0.2]], "prizes": [-0.1]}'
        result = parse_dataset(content, 'json', 'op')
        assert result is None

    def test_txt_still_works_for_tsp(self):
        """TXT 格式仍可用于 TSP"""
        result = parse_dataset("0.1 0.2\n0.3 0.4", 'txt', 'tsp')
        assert result is not None
        assert result['coordinates'] == [[0.1, 0.2], [0.3, 0.4]]

    def test_depot_optional_field_parsed(self):
        """depot 字段可选，解析后保留"""
        content = '{"coordinates": [[0.1, 0.2]], "depot": [0.5, 0.5]}'
        result = parse_dataset(content, 'json', 'cvrp')
        assert result is not None
        assert result['depot'] == [0.5, 0.5]
