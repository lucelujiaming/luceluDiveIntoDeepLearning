# 考虑下⾯这段简单的命令式程序：
def add(a, b):
	return a + b
def fancy_func(a, b, c, d):
	e = add(a, b)
	f = add(c, d)
	g = add(e, f)
	return g
print(fancy_func(1, 2, 3, 4))

# 下⾯，我们将通过模拟命令式编程来进⼀步了解符号式编程的概念。
#     符号式编程运⾏效率更⾼，更易于移植。
#     符号式编程更容易在编译期间优化代码，
#     同时还能够将程序移植到与Python⽆关的格式中，
#     从⽽允许程序在⾮Python环境中运⾏，
#     避免了任何潜在的与Python解释器相关的性能问题。
def add_():
	return '''
def add(a, b):
	return a + b
'''

def fancy_func_():
	return '''
def fancy_func(a, b, c, d):
	e = add(a, b)
	f = add(c, d)
	g = add(e, f)
	return g
'''
def evoke_():
	return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'
prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)

