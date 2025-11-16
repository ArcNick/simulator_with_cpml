# 直接编译运行
```bash
nvcc -rdc=true -I include src/*.cu -o bin/main_debug -std=c++17
./bin/main_debug
```

# 代码检测
```bash
compute-sanitizer --tool memcheck --leak-check full --log-file report.txt ./bin/main_debug
```