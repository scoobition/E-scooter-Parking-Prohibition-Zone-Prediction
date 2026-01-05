from data import set_data
from dbscan import apply_dbscan
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def visualize_results(df, title, save_path, x_col='경도', y_col='위도', label_col=None, plot_type='scatter'):
    """
    데이터프레임의 결과를 시각화합니다.

    :param df: pandas DataFrame (시각화할 데이터)
    :param title: str (그래프 제목)
    :param save_path: str (저장할 파일 경로)
    :param x_col: str (x축으로 사용할 컬럼 이름)
    :param y_col: str (y축으로 사용할 컬럼 이름)
    :param label_col: str, optional (색상 구분에 사용할 컬럼 이름)
    :param plot_type: str ('scatter' 또는 'path')
    """
    
    plt.figure(figsize=(12, 8))
    plot_title = title

    # 경로 시각화 (예: A* 알고리즘 결과)
    if plot_type == 'path':
        if not df.empty:
            # 경로를 선으로 연결
            plt.plot(df[x_col], df[y_col], 'o-', label='Path', color='red', markersize=5)
            # 시작점과 끝점 표시
            plt.plot(df[x_col].iloc[0], df[y_col].iloc[0], 'go', markersize=12, label='Start')
            plt.plot(df[x_col].iloc[-1], df[y_col].iloc[-1], 'bo', markersize=12, label='End')
            plt.legend()
    
    # 산점도 시각화 (예: 클러스터링 결과)
    elif plot_type == 'scatter':
        # label_col이 제공된 경우, 레이블에 따라 색상 구분
        if label_col and label_col in df.columns:
            labels = df[label_col]
            unique_labels = set(labels)
            
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            if n_clusters > 0:
                colors = plt.cm.get_cmap('Spectral', n_clusters)
            else:
                colors = None

            # 노이즈 포인트 (-1)
            noise_mask = (labels == -1)
            if np.any(noise_mask):
                xy_noise = df[noise_mask]
                plt.plot(xy_noise[x_col], xy_noise[y_col], 'o', markerfacecolor='gray',
                         markeredgecolor='k', markersize=3, label='Noise')

            # 클러스터링된 포인트
            if n_clusters > 0:
                cluster_labels = sorted([l for l in unique_labels if l != -1])
                for i, k in enumerate(cluster_labels):
                    class_member_mask = (labels == k)
                    xy = df[class_member_mask]
                    plt.plot(xy[x_col], xy[y_col], 'o', markerfacecolor=colors(i),
                             markeredgecolor='k', markersize=6, label=f'Cluster {k}')
            
            plt.legend()
            plot_title = f'{title} (클러스터 수: {n_clusters})'

        else:
            # 레이블 없이 모든 점을 동일하게 표시
            plt.plot(df[x_col], df[y_col], 'o', color='blue')
    
    else:
        print(f"Error: Unsupported plot_type '{plot_type}'.")
        return

    plt.title(plot_title)
    plt.xlabel(f'{x_col}')
    plt.ylabel(f'{y_col}')
    plt.grid(True)
    
    plt.savefig(save_path)
    print(f"\n그래프가 '{save_path}'에 저장되었습니다.")
    # plt.show() # 로컬에서 직접 실행할 경우 주석 해제

def main():
    print("Which work are you going to do?(press 6 to exit)")
    print("1. 데이터 로드")
    print("2. DB SCAN")
    print("3. A# ALGORITHM")
    print("4. VISUALIZING (DBSCAN)")
    print("5. ALL")
    command = input("Please enter the number of the work: ")
    while(command != "6"):
        if command == "1":
            set_data()
            command = input("Please enter the number of the work: ")
        elif command == "2":
            # 1. 데이터 로드
            print("데이터를 로드합니다...")
            df = pd.read_csv("data/processed_data.csv")

            # 2. DBSCAN 클러스터링 적용
            print("\nDBSCAN 클러스터링을 시작합니다...")
            result = apply_dbscan(df)
            
            # 3. 결과 출력
            print("\n클러스터링 결과:")
            print(result)
            print(f"\n총 {len(result)}개의 데이터 포인트 처리 완료.")
            print(f"발견된 클러스터 수 (노이즈 제외): {len(result[result['dbscan_label'] != -1]['dbscan_label'].unique())}")
            command = input("Please enter the number of the work: ")
        elif command == "3":
            print("This function is not exist yet")
            # 참고: 여기에 A* 알고리즘 구현 후 결과(df_path)를 생성한다고 가정
            # 예시: 
            # df_path = a_star_algorithm() 
            # visualize_results(df_path, title='A* Algorithm Path', save_path='data/a_star_path.png', plot_type='path')
            command = input("Please enter the number of the work: ")
        elif command == "4":
            try:
                result = pd.read_csv("data/dbscan.csv")
                # 결과 시각화
                print("\n결과를 산점도로 시각화합니다...")
                visualize_results(
                    result, 
                    title='DBSCAN Clustering Result', 
                    save_path='data/cluster_plot.png',
                    label_col='dbscan_label',
                    plot_type='scatter'
                )
            except FileNotFoundError:
                print("\n'data/dbscan.csv' 파일을 찾을 수 없습니다. 먼저 2번 DB SCAN을 실행하세요.")

            command = input("Please enter the number of the work: ")
        elif command == "5":
            set_data()
            # 1. 데이터 로드
            print("데이터를 로드합니다...")
            df = pd.read_csv("data/set_data.csv")

            # 2. DBSCAN 클러스터링 적용
            print("\nDBSCAN 클러스터링을 시작합니다...")
            result = apply_dbscan(df)

            # 3. 결과 출력
            print("\n클러스터링 결과:")
            print(result)
            print(f"\n총 {len(result)}개의 데이터 포인트 처리 완료.")
            print(f"발견된 클러스터 수 (노이즈 제외): {len(result[result['dbscan_label'] != -1]['dbscan_label'].unique())}")

            # 결과 시각화
            print("\n결과를 산점도로 시각화합니다...")
            visualize_results(
                result, 
                title='DBSCAN Clustering Result', 
                save_path='data/cluster_plot.png',
                label_col='dbscan_label',
                plot_type='scatter'
            )
            command = input("Please enter the number of the work: ")

if __name__ == "__main__":
    main()
