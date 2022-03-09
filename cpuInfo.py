import psutil


def cpuInfo():
    def memoryInfo(memory):
        # print(memory)
        return {
            'Total Memory': str(round((float(memory.total) / 1024 / 1024 / 1024), 2)) + "G",
            'Used Memory': str(round((float(memory.used) / 1024 / 1024 / 1024), 2)) + "G",
            'Free Memory': str(round((float(memory.free) / 1024 / 1024 / 1024), 2)) + "G",
            'Memory Percent': str(memory.percent) + '%'
        }

    print({'CPU': {
        'Physical Count': psutil.cpu_count(logical=False),
        'Logical Count': psutil.cpu_count(),
        'Percent': psutil.cpu_percent(interval=1),
        'Usage': psutil.cpu_percent(percpu=True, interval=1)},
        'Virtual Memory': memoryInfo(psutil.virtual_memory()),
        'Swap Memory': memoryInfo(psutil.swap_memory()),
        'GPU': 'None'
    })
    # 'GPU': {'Name: Tesla C2070', 'Total Memory: 5636292608', 'Free Memory: 5578420224',
    #         'Used Memory: 57872384'}


if __name__ == '__main__':
    cpuInfo()
