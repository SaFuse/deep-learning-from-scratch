# coding: utf-8
class Man:
    """サンプルクラス"""

    def __init__(self, name):
        self.name = name
        print("Initilized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")

m = Man("David")
m.hello()
m.goodbye()

class Prism:

	def __init__(self, width, height, depth):
		self.width = width
		self.height = height
		self.depth = depth

	def content(self):
		return self.width * self.height * self.depth

p1 = Prism(10, 20, 30)
print(p1.content())