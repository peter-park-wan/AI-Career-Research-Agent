"""Web 服务层（应用外壳）。

本包与 Agent 算法包 ``src/open_deep_research/`` 物理分离，边界如下：

- ``open_deep_research``：纯 Python Agent 库。它**不知道 HTTP 的存在**，
  只暴露 LangGraph 图、工作流函数与工具。
- ``app``：Web 服务层。负责协议、鉴权、配置、错误、日志、任务调度、持久化，
  通过调用 ``open_deep_research`` 的函数完成业务。

为什么要分这两个包：算法要能被 Studio、CLI、脚本、单元测试直接调用；
服务层要能独立演进（换框架、加中间件、做多副本），互不牵连。
"""
