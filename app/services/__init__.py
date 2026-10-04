"""业务服务层：编排领域逻辑，不依赖 FastAPI。

注意：这里**不做**子模块的 eager 重导出。服务之间（如 research_service → messages）
存在互相引用，若 ``__init__`` 提前 import 全部子模块，package 初始化期间会形成
部分初始化循环，报错信息还很难读。需要什么就直接
``from app.services.xxx import yyy``。
"""
