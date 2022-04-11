import os
import subprocess
import sys
import time

if __name__ == '__main__':
    while True:
        result = os.popen('ps -all|grep axel')
        msg = result.buffer.read().decode('utf-8')
        lines = msg.splitlines()
        # 如果没有 axel 进程了
        if len(lines) == 0:
            # 如果文件已经下载完了，就 exit(0)
            with open("/data/hky/UNITER/download_vqa.log", 'r') as f:
                lines = f.readlines()  # 读取所有行
                last_line = lines[-1]  # 取最后一行
                before_last_line = lines[-2]  # 取倒数第二行
                if '100%' in last_line or '100%' in before_last_line:
                    print("\ndownload finished!")
                    exit(0)

            # 没下载完说明中断了进程需要重启
            re_code = subprocess.call(
                "nohup axel -n 64 https://acvrpublicycchen.blob.core.windows.net/uniter/img_db/coco_test2015.tar -o ~/processed_data_and_pretrained_models/img_db/  >/data/hky/UNITER/download_vqa.log &",
                shell=True)
            if re_code == 0:
                print("\nstart axel successfully! at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            else:
                print("\nfailed to start axel!")
        else:
            print('.', end='')
        sys.stdout.flush()
        time.sleep(5)
