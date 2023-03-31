from selenium import webdriver
import schedule
import time
from bs4 import BeautifulSoup
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

options = webdriver.ChromeOptions()
options.add_argument('headless')
driver = webdriver.Chrome('./chromedriver',options=options)
url = 'https://finance.daum.net/quotes/A005930?period=day#home'
driver.get(url)

x_val=[]
y_val=[]

count = 0

def real_time():

    global x_val
    global y_val
  
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    test1 = soup.select('#boxSummary > div > span:nth-child(1) > span.currentB > span.numB > strong')
   
    string = str(test1)
    String_data = string[41:47] #삼성전자 시가
    now = datetime.datetime.now()
    nowTime = now.strftime('%H:%M:%S') #현재시간
    print("")
    print('*******************************************')
    print("현재시간 삼성전자 주가 : "+nowTime+"--> ",String_data)
    print('*******************************************')
    print("")

    #print(type(nowTime))
    y_val.append(String_data)
    x_val.append(nowTime)
    
    #print(x_val)

schedule.every(2).seconds.do(real_time)

for i in range(0,10000):
    
    print(x_val)
    print(y_val)
    
    schedule.run_pending()
    time.sleep(1)
    count = count + 1
        
    
    if count > 20:
        raw_data = { 'time' : x_val,
              'stock_price' : y_val   }
    
        raw_data = pd.DataFrame(raw_data)
        raw_data.to_excel(excel_writer='C://test_data//test_stock2.xlsx')  
        
        break
    
def animate(i):
    
    plt.cla()
    plt.plot(x_val,y_val)
    plt.xlabel('time')
    plt.ylabel('stock_price')
    plt.title('Fluctuations of stock price')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
        
ani = FuncAnimation(plt.gcf(), animate, 1000)
plt.tight_layout()
plt.show()

'''
while True:      
    schedule.run_pending()
    time.sleep(1)
'''    

