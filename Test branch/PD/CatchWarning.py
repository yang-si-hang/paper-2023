import trace
import pdb
import warnings

class RuntimeWarningTracer(trace.Trace):

    def globaltrace(self, frame, event, arg):
        # 当函数调用发生时
        if event == "call":
            # 检查是否调用了 _showwarning 函数
            if frame.f_code.co_name == "_showwarning":
                category = frame.f_locals['category']
                if category is RuntimeWarning:
                    pdb.set_trace()
        return self.localtrace

    def localtrace(self, frame, event, arg):
        return self.localtrace

# 将新的警告过滤器设置为 "always"
warnings.simplefilter('always', RuntimeWarning)

with open('DiffPD.py', 'rb') as file:
    code_content = file.read()

# 启动跟踪器
tracer = RuntimeWarningTracer()
tracer.run(code_content)  # 替换 YOUR_CODE_HERE() 为您要执行的代码或函数
