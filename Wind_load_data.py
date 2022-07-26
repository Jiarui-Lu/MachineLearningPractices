import pandas as pd
from WindPy import w
import datetime

w.start()
w.isconnected()


# Wind导入数据的demo程序
class load_data(object):
    def __init__(self, fund_codes, index_codes, start, end):
        # 代码，起始日、结束日
        self.fund_codes = fund_codes
        self.index_codes = index_codes
        self.start, self.end = start, end

    def load_fund(self):
        error_code, data = w.wsd(self.fund_codes, "nav", self.start, self.end, "", usedf=True)
        return data

    def load_index(self):
        error_code, data = w.wsd(self.index_codes, "close", self.start, self.end, "", usedf=True)
        return data


if __name__ == '__main__':
    fund_codes = "000082.OF,000309.OF,000326.OF,000409.OF"
    index_codes = "399314.SZ,399316.SZ,399370.SZ,399371.SZ,000001.SH,399001.SZ"
    start = '2021-07-16'
    end = '2022-07-15'
    x = load_data(fund_codes, index_codes, start, end)
    # print(x.load_index())
    # print(x.load_fund())
    # pd.DataFrame.to_excel(x.load_fund())
    # pd.DataFrame.to_excel(x.load_index())
    print('导出完成')
