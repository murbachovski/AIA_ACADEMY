class FourCal:
    def __init__(self, first, second): # 예약어, 이거를 일단 먼저 실행한다.
        self.first = first
        self.second = second
    def setdata(self, first, second):
        self.first = first
        self.second = second
    def add(self):
        result = self.first + self.second
        return result
    
a = FourCal(1,2)

#23.04.20 아침에 Class공부