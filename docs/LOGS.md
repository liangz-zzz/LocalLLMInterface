# 日志管理说明

Local LLM Interface 现在支持完整的日志管理功能，包括文件日志和便捷的查看脚本。

## 📁 日志文件位置

```
LocalLLMInterface/
├── logs/
│   ├── app.log              # 应用程序主日志文件
│   ├── app.log.2025-08-30   # 轮转的历史日志文件
│   └── app.log.zip          # 压缩的旧日志文件
└── view-logs.sh             # 日志查看脚本
```

## 🔧 日志配置特性

### 双重日志输出
- **控制台**: 彩色格式，便于开发调试
- **文件**: 纯文本格式，便于分析和存档

### 自动轮转和清理
- **文件大小**: 单文件最大 100MB
- **保留时间**: 7天自动清理
- **压缩存储**: 旧文件自动压缩为 `.zip`

### Docker容器日志
- **格式**: JSON格式，便于日志分析工具处理
- **轮转**: 最多保留5个文件，每个100MB

## 🛠️ 使用日志查看脚本

### 基本用法
```bash
# 查看帮助
./view-logs.sh help

# 查看应用日志
./view-logs.sh app

# 实时跟踪日志
./view-logs.sh follow

# 查看错误信息
./view-logs.sh errors
```

### 所有可用选项

| 命令 | 说明 |
|------|------|
| `app`, `application` | 查看应用程序日志文件 |
| `docker`, `container` | 查看Docker容器日志 |
| `all`, `both` | 同时显示应用和Docker日志 |
| `tail`, `follow` | 实时跟踪应用日志 |
| `errors`, `error` | 过滤显示错误和警告信息 |
| `clear`, `clean` | 清空应用日志文件 |
| `help` | 显示帮助信息 |

## 📊 日志格式说明

### 应用程序日志格式
```
2025-08-30 15:59:32 | INFO     | app.main:lifespan:40 - Starting Local LLM Interface
时间戳              | 级别     | 模块:函数:行号        - 消息内容
```

### 日志级别
- `DEBUG`: 详细调试信息
- `INFO`: 一般信息（默认级别）  
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误

## 🔍 常用日志查看场景

### 1. 服务启动检查
```bash
# 查看最近的启动日志
./view-logs.sh app | tail -20

# 或查看特定时间段
grep "2025-08-30 15:59" logs/app.log
```

### 2. 错误排查
```bash
# 查看所有错误
./view-logs.sh errors

# 查找特定错误
grep -i "failed\|exception" logs/app.log
```

### 3. 模型加载监控
```bash
# 实时监控模型切换
./view-logs.sh follow | grep -i "loading\|switching\|unloading"
```

### 4. API请求监控
```bash
# 查看API请求日志
./view-logs.sh follow | grep -E "POST|GET"
```

## 🚨 故障排除

### 日志文件不存在
如果 `logs/app.log` 不存在：
1. 确认容器正在运行: `docker ps`
2. 检查容器日志: `docker logs local-llm-api`  
3. 重启服务: `docker-compose restart`

### 权限问题
如果无法创建日志文件：
```bash
# 确保logs目录有正确权限
sudo chown -R $USER:$USER logs/
chmod 755 logs/
```

### 日志脚本无法执行
```bash
# 添加执行权限
chmod +x view-logs.sh
```

## 💡 最佳实践

1. **开发期间**: 使用 `./view-logs.sh follow` 实时监控
2. **生产环境**: 定期检查 `./view-logs.sh errors`
3. **性能分析**: 结合GPU监控 `nvidia-smi` 和日志分析
4. **存储管理**: 日志会自动轮转，但可手动清理 `./view-logs.sh clear`

## 🔧 日志级别调整

在 `docker-compose.yml` 中修改日志级别：

```yaml
environment:
  - LLM_LOG_LEVEL=DEBUG  # 详细日志
  - LLM_LOG_LEVEL=INFO   # 标准日志(默认)
  - LLM_LOG_LEVEL=WARNING # 只显示警告和错误
```

重启服务生效：
```bash
docker-compose restart
```

---

现在您可以方便地在项目目录下直接查看和管理所有日志信息！ 🎉