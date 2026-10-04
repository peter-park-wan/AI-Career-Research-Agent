"""HTTP 路由层：只做参数绑定、鉴权声明与响应组装。

硬性约束（可用于 code review / 静态检查）：
* 不出现裸文件 IO（``open`` / ``Path.read_text``）
* 不出现业务规则分支（业务判断属于 ``app/services``）
* 不出现 ``os.getenv``
"""

__all__ = ["career", "health", "profile", "research"]
