import streamlit as st
import pyrebase
import time
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------
# Firebase 설정
# ---------------------
firebase_config = {
    "apiKey": "AIzaSyCswFmrOGU3FyLYxwbNPTp7hvQxLfTPIZw",
    "authDomain": "sw-projects-49798.firebaseapp.com",
    "databaseURL": "https://sw-projects-49798-default-rtdb.firebaseio.com",
    "projectId": "sw-projects-49798",
    "storageBucket": "sw-projects-49798.firebasestorage.app",
    "messagingSenderId": "812186368395",
    "appId": "1:812186368395:web:be2f7291ce54396209d78e"
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
firestore = firebase.database()
storage = firebase.storage()

# ---------------------
# 세션 상태 초기화
# ---------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.session_state.id_token = ""
    st.session_state.user_name = ""
    st.session_state.user_gender = "선택 안함"
    st.session_state.user_phone = ""
    st.session_state.profile_image_url = ""

# ---------------------
# 홈 페이지 클래스
# ---------------------
class Home:
    def __init__(self, login_page, register_page, findpw_page):
        st.title("🏠 Home")
        if st.session_state.get("logged_in"):
            st.success(f"{st.session_state.get('user_email')}님 환영합니다.")

        # Kaggle 데이터셋 출처 및 소개
        st.markdown("""
                ---
                **Bike Sharing Demand 데이터셋**  
                - 제공처: [Kaggle Bike Sharing Demand Competition](https://www.kaggle.com/c/bike-sharing-demand)  
                - 설명: 2011–2012년 캘리포니아 주의 수도인 미국 워싱턴 D.C. 인근 도시에서 시간별 자전거 대여량을 기록한 데이터  
                - 주요 변수:  
                  - `datetime`: 날짜 및 시간  
                  - `season`: 계절  
                  - `holiday`: 공휴일 여부  
                  - `workingday`: 근무일 여부  
                  - `weather`: 날씨 상태  
                  - `temp`, `atemp`: 기온 및 체감온도  
                  - `humidity`, `windspeed`: 습도 및 풍속  
                  - `casual`, `registered`, `count`: 비등록·등록·전체 대여 횟수  
                """)

# ---------------------
# 로그인 페이지 클래스
# ---------------------
class Login:
    def __init__(self):
        st.title("🔐 로그인")
        email = st.text_input("이메일")
        password = st.text_input("비밀번호", type="password")
        if st.button("로그인"):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.id_token = user['idToken']

                user_info = firestore.child("users").child(email.replace(".", "_")).get().val()
                if user_info:
                    st.session_state.user_name = user_info.get("name", "")
                    st.session_state.user_gender = user_info.get("gender", "선택 안함")
                    st.session_state.user_phone = user_info.get("phone", "")
                    st.session_state.profile_image_url = user_info.get("profile_image_url", "")

                st.success("로그인 성공!")
                time.sleep(1)
                st.rerun()
            except Exception:
                st.error("로그인 실패")

# ---------------------
# 회원가입 페이지 클래스
# ---------------------
class Register:
    def __init__(self, login_page_url):
        st.title("📝 회원가입")
        email = st.text_input("이메일")
        password = st.text_input("비밀번호", type="password")
        name = st.text_input("성명")
        gender = st.selectbox("성별", ["선택 안함", "남성", "여성"])
        phone = st.text_input("휴대전화번호")

        if st.button("회원가입"):
            try:
                auth.create_user_with_email_and_password(email, password)
                firestore.child("users").child(email.replace(".", "_")).set({
                    "email": email,
                    "name": name,
                    "gender": gender,
                    "phone": phone,
                    "role": "user",
                    "profile_image_url": ""
                })
                st.success("회원가입 성공! 로그인 페이지로 이동합니다.")
                time.sleep(1)
                st.switch_page(login_page_url)
            except Exception:
                st.error("회원가입 실패")

# ---------------------
# 비밀번호 찾기 페이지 클래스
# ---------------------
class FindPassword:
    def __init__(self):
        st.title("🔎 비밀번호 찾기")
        email = st.text_input("이메일")
        if st.button("비밀번호 재설정 메일 전송"):
            try:
                auth.send_password_reset_email(email)
                st.success("비밀번호 재설정 이메일을 전송했습니다.")
                time.sleep(1)
                st.rerun()
            except:
                st.error("이메일 전송 실패")

# ---------------------
# 사용자 정보 수정 페이지 클래스
# ---------------------
class UserInfo:
    def __init__(self):
        st.title("👤 사용자 정보")

        email = st.session_state.get("user_email", "")
        new_email = st.text_input("이메일", value=email)
        name = st.text_input("성명", value=st.session_state.get("user_name", ""))
        gender = st.selectbox(
            "성별",
            ["선택 안함", "남성", "여성"],
            index=["선택 안함", "남성", "여성"].index(st.session_state.get("user_gender", "선택 안함"))
        )
        phone = st.text_input("휴대전화번호", value=st.session_state.get("user_phone", ""))

        uploaded_file = st.file_uploader("프로필 이미지 업로드", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_path = f"profiles/{email.replace('.', '_')}.jpg"
            storage.child(file_path).put(uploaded_file, st.session_state.id_token)
            image_url = storage.child(file_path).get_url(st.session_state.id_token)
            st.session_state.profile_image_url = image_url
            st.image(image_url, width=150)
        elif st.session_state.get("profile_image_url"):
            st.image(st.session_state.profile_image_url, width=150)

        if st.button("수정"):
            st.session_state.user_email = new_email
            st.session_state.user_name = name
            st.session_state.user_gender = gender
            st.session_state.user_phone = phone

            firestore.child("users").child(new_email.replace(".", "_")).update({
                "email": new_email,
                "name": name,
                "gender": gender,
                "phone": phone,
                "profile_image_url": st.session_state.get("profile_image_url", "")
            })

            st.success("사용자 정보가 저장되었습니다.")
            time.sleep(1)
            st.rerun()

# ---------------------
# 로그아웃 페이지 클래스
# ---------------------
class Logout:
    def __init__(self):
        st.session_state.logged_in = False
        st.session_state.user_email = ""
        st.session_state.id_token = ""
        st.session_state.user_name = ""
        st.session_state.user_gender = "선택 안함"
        st.session_state.user_phone = ""
        st.session_state.profile_image_url = ""
        st.success("로그아웃 되었습니다.")
        time.sleep(1)
        st.rerun()

# ---------------------
# EDA 페이지 클래스
# ---------------------
class EDA:
    def __init__(self):
        st.title("📊 Bike Sharing Demand EDA")
        uploaded = st.file_uploader("데이터셋 업로드 (train.csv)", type="csv")
        if not uploaded:
            st.info("train.csv 파일을 업로드 해주세요.")
            return

        df = pd.read_csv(uploaded, parse_dates=['datetime'])

        tabs = st.tabs([
            "1. 목적 & 절차",
            "2. 데이터셋 설명",
            "3. 데이터 로드 & 품질 체크",
            "4. Datetime 특성 추출",
            "5. 시각화",
            "6. 상관관계 분석",
            "7. 이상치 제거",
            "8. 로그 변환"
        ])

        # 1. 목적 & 분석 절차
        with tabs[0]:
            st.header("🔭 목적 & 분석 절차")
            st.markdown("""
            **목적**: Bike Sharing Demand 데이터셋을 탐색하고,
            다양한 특성이 대여량(count)에 미치는 영향을 파악합니다.

            **절차**:
            1. 데이터 구조 및 기초 통계 확인  
            2. 결측치/중복치 등 품질 체크  
            3. datetime 특성(연도, 월, 일, 시, 요일) 추출  
            4. 주요 변수 시각화  
            5. 변수 간 상관관계 분석  
            6. 이상치 탐지 및 제거  
            7. 로그 변환을 통한 분포 안정화
            """)

        # 2. 데이터셋 설명
        with tabs[1]:
            st.header("🔍 데이터셋 설명")
            st.markdown(f"""
            - **train.csv**: 2011–2012년까지의 시간대별 대여 기록  
            - 총 관측치: {df.shape[0]}개  
            - 주요 변수:
              - **datetime**: 날짜와 시간 (YYYY-MM-DD HH:MM:SS)  
              - **season**: 계절 (1: 봄, 2: 여름, 3: 가을, 4: 겨울)  
              - **holiday**: 공휴일 여부 (0: 평일, 1: 공휴일)  
              - **workingday**: 근무일 여부 (0: 주말/공휴일, 1: 근무일)  
              - **weather**: 날씨 상태  
                - 1: 맑음·부분적으로 흐림  
                - 2: 안개·흐림  
                - 3: 가벼운 비/눈  
                - 4: 폭우/폭설 등  
              - **temp**: 실제 기온 (섭씨)  
              - **atemp**: 체감 온도 (섭씨)  
              - **humidity**: 상대 습도 (%)  
              - **windspeed**: 풍속 (정규화된 값)  
              - **casual**: 비등록 사용자 대여 횟수  
              - **registered**: 등록 사용자 대여 횟수  
              - **count**: 전체 대여 횟수 (casual + registered)
            """)

            st.subheader("1) 데이터 구조 (`df.info()`)")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

            st.subheader("2) 기초 통계량 (`df.describe()`)")
            numeric_df = df.select_dtypes(include=[np.number])
            st.dataframe(numeric_df.describe())

            st.subheader("3) 샘플 데이터 (첫 5행)")
            st.dataframe(df.head())

        # 3. 데이터 로드 & 품질 체크
        with tabs[2]:
            st.header("📥 데이터 로드 & 품질 체크")
            st.subheader("결측값 개수")
            missing = df.isnull().sum()
            st.bar_chart(missing)

            duplicates = df.duplicated().sum()
            st.write(f"- 중복 행 개수: {duplicates}개")

        # 4. Datetime 특성 추출
        with tabs[3]:
            st.header("🕒 Datetime 특성 추출")
            st.markdown("`datetime` 컬럼에서 연, 월, 일, 시, 요일 등을 추출합니다.")

            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['hour'] = df['datetime'].dt.hour
            df['dayofweek'] = df['datetime'].dt.dayofweek

            st.subheader("추출된 특성 예시")
            st.dataframe(df[['datetime', 'year', 'month', 'day', 'hour',
                             'dayofweek']].head())

            # --- 요일 숫자 → 요일명 매핑 (참고용) ---
            day_map = {
                0: '월요일',
                1: '화요일',
                2: '수요일',
                3: '목요일',
                4: '금요일',
                5: '토요일',
                6: '일요일'
            }
            st.markdown("**(참고) dayofweek 숫자 → 요일**")
            # 중복 제거 후 정렬하여 표시
            mapping_df = pd.DataFrame({
                'dayofweek': list(day_map.keys()),
                'weekday': list(day_map.values())
            })
            st.dataframe(mapping_df, hide_index=True)

        # 5. 시각화
        with tabs[4]:
            st.header("📈 시각화")
            # by 근무일 여부
            st.subheader("근무일 여부별 시간대별 평균 대여량")
            fig1, ax1 = plt.subplots()
            sns.pointplot(x='hour', y='count', hue='workingday', data=df,
                          ax=ax1)
            ax1.set_xlabel("Hour");
            ax1.set_ylabel("Average Count")
            st.pyplot(fig1)
            st.markdown(
                "> **해석:** 근무일(1)은 출퇴근 시간(7 ~ 9시, 17 ~ 19시)에 대여량이 급증하는 반면,\n"
                "비근무일(0)은 오후(11 ~ 15시) 시간대에 대여량이 상대적으로 높게 나타납니다."
            )

            # by 요일
            st.subheader("요일별 시간대별 평균 대여량")
            fig2, ax2 = plt.subplots()
            sns.pointplot(x='hour', y='count', hue='dayofweek', data=df, ax=ax2)
            ax2.set_xlabel("Hour");
            ax2.set_ylabel("Average Count")
            st.pyplot(fig2)
            st.markdown(
                "> **해석:** 평일(월 ~ 금)은 출퇴근 피크가 두드러지고,\n"
                "주말(토~일)은 오전 중반(10 ~ 14시)에 대여량이 더 고르게 분포하는 경향이 있습니다."
            )

            # by 시즌
            st.subheader("시즌별 시간대별 평균 대여량")
            fig3, ax3 = plt.subplots()
            sns.pointplot(x='hour', y='count', hue='season', data=df, ax=ax3)
            ax3.set_xlabel("Hour");
            ax3.set_ylabel("Average Count")
            st.pyplot(fig3)
            st.markdown(
                "> **해석:** 여름(2)과 가을(3)에 전반적으로 대여량이 높고,\n"
                "겨울(4)은 전 시간대에 걸쳐 대여량이 낮게 나타납니다."
            )

            # by 날씨
            st.subheader("날씨 상태별 시간대별 평균 대여량")
            fig4, ax4 = plt.subplots()
            sns.pointplot(x='hour', y='count', hue='weather', data=df, ax=ax4)
            ax4.set_xlabel("Hour");
            ax4.set_ylabel("Average Count")
            st.pyplot(fig4)
            st.markdown(
                "> **해석:** 맑음(1)은 전 시간대에서 대여량이 가장 높으며,\n"
                "안개·흐림(2), 가벼운 비/눈(3)에선 다소 감소하고, 심한 기상(4)에서는 크게 떨어집니다."
            )

        # 6. 상관관계 분석
        with tabs[5]:
            st.header("🔗 상관관계 분석")
            # 관심 피처만 선택
            features = ['temp', 'atemp', 'casual', 'registered', 'humidity',
                        'windspeed', 'count']
            corr_df = df[features].corr()

            # 상관계수 테이블 출력
            st.subheader("📊 피처 간 상관계수")
            st.dataframe(corr_df)

            # 히트맵 시각화
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_xlabel("")  # 축 이름 제거
            ax.set_ylabel("")
            st.pyplot(fig)
            st.markdown(
                "> **해석:**\n"
                "- `count`는 `registered` (r≈0.99) 및 `casual` (r≈0.67)와 강한 양의 상관관계를 보입니다.\n"
                "- `temp`·`atemp`와 `count`는 중간 정도의 양의 상관관계(r≈0.4~0.5)를 나타내며, 기온이 높을수록 대여량이 증가함을 시사합니다.\n"
                "- `humidity`와 `windspeed`는 약한 음의 상관관계(r≈-0.2~-0.3)를 보여, 습도·풍속이 높을수록 대여량이 다소 감소합니다."
            )

        # 7. 이상치 제거
        with tabs[6]:
            st.header("🚫 이상치 제거")
            # 평균·표준편차 계산
            mean_count = df['count'].mean()
            std_count = df['count'].std()
            # 상한치: 평균 + 3*표준편차
            upper = mean_count + 3 * std_count

            st.markdown(f"""
                        - **평균(count)**: {mean_count:.2f}  
                        - **표준편차(count)**: {std_count:.2f}  
                        - **이상치 기준**: `count` > 평균 + 3×표준편차 = {upper:.2f}  
                          (통계학의 68-95-99.7 법칙(Empirical rule)에 따라 평균에서 3σ를 벗어나는 관측치는 전체의 약 0.3%로 극단치로 간주)
                        """)
            df_no = df[df['count'] <= upper]
            st.write(f"- 이상치 제거 전: {df.shape[0]}개, 제거 후: {df_no.shape[0]}개")

        # 8. 로그 변환
        with tabs[7]:
            st.header("🔄 로그 변환")
            st.markdown("""
                **로그 변환 맥락**  
                - `count` 변수는 오른쪽으로 크게 치우친 분포(skewed distribution)를 가지고 있어,  
                  통계 분석 및 모델링 시 정규성 가정이 어렵습니다.  
                - 따라서 `Log(Count + 1)` 변환을 통해 분포의 왜도를 줄이고,  
                  중앙값 주변으로 데이터를 모아 해석력을 높입니다.
                """)
                # -----------------------
        # 인구 추이 분석 탭 추가
        # -----------------------
        st.markdown("---")
        st.header("📈 인구 추이 데이터 분석")
        pop_file = st.file_uploader("인구 데이터 업로드 (population_trends.csv)", type="csv", key="population")
        if pop_file:
            pop_df = pd.read_csv(pop_file)

            # '-' 처리 및 세종 결측치 → 0 치환
            pop_df.replace("-", np.nan, inplace=True)
            pop_df.loc[pop_df['지역'] == "세종", ['인구', '출생아수(명)', '사망자수(명)']] = \
                pop_df.loc[pop_df['지역'] == "세종", ['인구', '출생아수(명)', '사망자수(명)']].replace(np.nan, 0)
            
            # 숫자형 변환
            for col in ['인구', '출생아수(명)', '사망자수(명)']:
                pop_df[col] = pd.to_numeric(pop_df[col], errors='coerce')

            # 탭 구성
            pop_tabs = st.tabs([
                "1. 결측치 및 중복 확인",
                "2. 연도별 전체 인구 추이 그래프",
                "3. 지역별 인구 변화량 순위",
                "4. 증감률 상위 지역 및 연도 도출",
                "5. 누적영역그래프 등 적절한 시각화"
            ])

            # 1. 결측치 및 중복 확인
            with pop_tabs[0]:
                st.subheader("1. 결측치 및 중복 확인")
                st.write("컬럼별 결측치")
                st.bar_chart(pop_df.isnull().sum())

                dup_count = pop_df.duplicated().sum()
                st.write(f"- 중복 행 개수: {dup_count}개")

                st.subheader("데이터 샘플")
                st.dataframe(pop_df.head())

            # 2. 연도별 전체 인구 추이 그래프
            with pop_tabs[1]:
                st.subheader("2. 연도별 전체 인구 추이 그래프")
                national = pop_df[pop_df['지역'] == "전국"].sort_values("연도")
                fig, ax = plt.subplots()
                sns.lineplot(data=national, x="연도", y="인구", marker="o", ax=ax)
                ax.set_title("Population Trend (Nationwide)")
                ax.set_xlabel("Year")
                ax.set_ylabel("Population")

                # 2035년 예측
                recent = national.tail(3)
                delta = (recent["출생아수(명)"] - recent["사망자수(명)"]).mean()
                last_year = national["연도"].max()
                last_pop = national["인구"].iloc[-1]
                forecast_year = 2035
                forecast_pop = int(last_pop + delta * (forecast_year - last_year))
                ax.axvline(forecast_year, color='gray', linestyle='--')
                ax.scatter(forecast_year, forecast_pop, color='red')
                ax.text(forecast_year, forecast_pop, f"{forecast_pop:,}", ha='left', va='bottom')
                st.pyplot(fig)

            # 3. 지역별 인구 변화량 순위
            with pop_tabs[2]:
                st.subheader("3. 지역별 인구 변화량 순위")
                recent_years = sorted(pop_df['연도'].unique())[-5:]
                regional = pop_df[(pop_df['연도'].isin(recent_years)) & (pop_df['지역'] != "전국")].copy()
                pivot = regional.pivot(index='연도', columns='지역', values='인구')
                delta = (pivot.loc[recent_years[-1]] - pivot.loc[recent_years[0]]) / 1000
                delta_sorted = delta.sort_values(ascending=False)

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x=delta_sorted.values, y=delta_sorted.index, ax=ax, palette="crest")
                ax.set_title("5-Year Population Change by Region")
                ax.set_xlabel("Change (thousands)")
                ax.bar_label(ax.containers[0], fmt="%.0f", padding=3)
                st.pyplot(fig)

            # 4. 증감률 상위 지역 및 연도 도출
            with pop_tabs[3]:
                st.subheader("4. 증감률 상위 지역 및 연도 도출")
                pivot = pop_df.pivot(index="연도", columns="지역", values="인구")
                pct_change = pivot.pct_change().dropna() * 100
                mean_change = pct_change.mean().sort_values(ascending=False)

                st.write("연평균 증감률 상위 지역")
                st.dataframe(mean_change.head(10).reset_index().rename(columns={0: "평균 증감률(%)"}))

                st.write("연도별 증감률 히트맵")
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(pct_change, annot=True, fmt=".1f", cmap="coolwarm", center=0, ax=ax)
                ax.set_title("Yearly % Change by Region")
                st.pyplot(fig)

            # 5. 누적영역그래프 등 적절한 시각화
            with pop_tabs[4]:
                st.subheader("5. 누적영역그래프 등 적절한 시각화")
                df_area = pop_df[pop_df['지역'] != "전국"]
                pivot_area = df_area.pivot(index="연도", columns="지역", values="인구")
                pivot_area.fillna(0, inplace=True)

                fig, ax = plt.subplots(figsize=(12, 6))
                pivot_area.plot.area(ax=ax, cmap="tab20", stacked=True)
                ax.set_title("Stacked Area Chart by Region")
                ax.set_ylabel("Population")
                ax.set_xlabel("Year")
                st.pyplot(fig)


# ---------------------
# 페이지 객체 생성
# ---------------------
Page_Login    = st.Page(Login,    title="Login",    icon="🔐", url_path="login")
Page_Register = st.Page(lambda: Register(Page_Login.url_path), title="Register", icon="📝", url_path="register")
Page_FindPW   = st.Page(FindPassword, title="Find PW", icon="🔎", url_path="find-password")
Page_Home     = st.Page(lambda: Home(Page_Login, Page_Register, Page_FindPW), title="Home", icon="🏠", url_path="home", default=True)
Page_User     = st.Page(UserInfo, title="My Info", icon="👤", url_path="user-info")
Page_Logout   = st.Page(Logout,   title="Logout",  icon="🔓", url_path="logout")
Page_EDA      = st.Page(EDA,      title="EDA",     icon="📊", url_path="eda")

# ---------------------
# 네비게이션 실행
# ---------------------
if st.session_state.logged_in:
    pages = [Page_Home, Page_User, Page_Logout, Page_EDA]
else:
    pages = [Page_Home, Page_Login, Page_Register, Page_FindPW]

selected_page = st.navigation(pages)
selected_page.run()