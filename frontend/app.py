import streamlit as st
import requests
import cv2
import numpy as np
from datetime import datetime
import json
import base64
import os
from dotenv import load_dotenv
from loguru import logger # ロギングを追加

# .env ファイルから環境変数を読み込む
load_dotenv()

# --- 設定 ---
# バックエンドAPIのURLとAPIキーを環境変数から取得
# 環境変数が設定されていない場合のデフォルト値も設定可能
API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
API_KEY = os.getenv("BACKEND_API_KEY", "YOUR_DEFAULT_API_KEY_HERE") # デフォルトキーを設定するか、Noneにしてエラー処理

# Streamlitログイン用の認証情報 (デモ用)
ADMIN_USER = os.getenv("ADMIN", "admin")
ADMIN_PASSWORD = os.getenv("PASSWORD", "password")

# ロガー設定 (オプション)
logger.add(lambda msg: print(msg, end=""), level="INFO")

# APIリクエスト用のヘッダー
HEADERS = {"X-API-KEY": API_KEY}

def init_user_data():
    if "users" not in st.session_state:
        st.session_state.users = {
            "admin": {
                "password": "password",
                "email": "admin@example.com",
                "created_at": datetime.now().isoformat()
            }
        }

# --- ログインページ ---
def login_page():
    """Display login page and handle authentication"""
    init_user_data()
    st.title("Welcome to YOLO Detection App")
    st.write("Please sign in to continue")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")

        if submitted:
            if username in st.session_state.users and st.session_state.users[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                logger.info(f"User '{username}' logged in successfully.")
                st.success("Login successful!")
                st.rerun()
            else:
                logger.warning(f"Failed login attempt for username: '{username}'")
                st.error("Invalid username or password")

    st.write("---")
    st.write("Don't have an account?")
    if st.button("Register here"):
        st.session_state.page = "register"
        st.rerun()
        
# --- 登録ページ ---
def register_page():
    """Display registration page"""
    init_user_data()
    st.title("Create New Account")

    with st.form("register_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        email = st.text_input("Email")
        submitted = st.form_submit_button("Register")

        if submitted:
            if not username or not password:
                st.error("Please fill in all required fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            elif username in st.session_state.users:
                st.error("Username already exists")
            else:
                # ユーザー情報をセッションに保存
                st.session_state.users[username] = {
                    "password": password,
                    "email": email,
                    "created_at": datetime.now().isoformat()
                }
                logger.info(f"Registration successful for username: '{username}'")
                st.success("Registration successful! Please sign in.")
                st.session_state.page = "login"
                st.rerun()

    st.write("---")
    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()
        
def user_account_page():
    """User account management page"""
    st.title("Account Settings")
    
    # ユーザー情報の取得
    try:
        response = requests.get(
            f"{API_URL}/users/me",
            headers={"Authorization": f"Bearer {st.session_state.get('access_token')}"}
        )
        if response.status_code == 200:
            user_data = response.json()
            st.write(f"### Welcome, {user_data['username']}")
            
            # アカウント情報の表示
            st.subheader("Account Information")
            st.write(f"Username: {user_data['username']}")
            st.write(f"Account created: {user_data['created_at']}")
            
            # パスワード変更フォーム
            st.subheader("Change Password")
            with st.form("change_password"):
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_new_password = st.text_input("Confirm New Password", type="password")
                if st.form_submit_button("Update Password"):
                    if new_password != confirm_new_password:
                        st.error("New passwords do not match")
                    else:
                        try:
                            response = requests.post(
                                f"{API_URL}/users/change-password",
                                json={
                                    "current_password": current_password,
                                    "new_password": new_password
                                },
                                headers={"Authorization": f"Bearer {st.session_state.get('access_token')}"}
                            )
                            if response.status_code == 200:
                                st.success("Password updated successfully!")
                            else:
                                st.error("Failed to update password")
                        except Exception as e:
                            st.error(f"Error updating password: {str(e)}")
            
            # アカウント削除オプション
            st.subheader("Delete Account")
            if st.button("Delete Account", type="primary"):
                if st.checkbox("I understand this action cannot be undone"):
                    try:
                        response = requests.delete(
                            f"{API_URL}/users/me",
                            headers={"Authorization": f"Bearer {st.session_state.get('access_token')}"}
                        )
                        if response.status_code == 200:
                            st.session_state.logged_in = False
                            st.session_state.access_token = None
                            st.success("Account deleted successfully")
                            st.rerun()
                        else:
                            st.error("Failed to delete account")
                    except Exception as e:
                        st.error(f"Error deleting account: {str(e)}")
    
    except Exception as e:
        st.error(f"Error loading user data: {str(e)}")

# --- メインアプリケーションページ ---
def main_app():
    """Main application after successful login"""
    st.set_page_config(page_title="YOLO Object Detection", page_icon="🔍", layout="wide")

    # ヘッダーとログアウトボタン
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("YOLO Object Detection")
    with col2:
        if st.button("Sign Out"):
            logger.info(f"User '{st.session_state.username}' logged out.")
            st.session_state.logged_in = False
            st.session_state.username = None
            # 必要に応じて他のセッション状態もクリア
            if 'feedback_data' in st.session_state:
                del st.session_state.feedback_data
            st.rerun() # ログインページへ

    st.write(f"Welcome, {st.session_state.username}!")

    # サイドバーにアカウント管理メニューを追加
    with st.sidebar:
        st.title("Menu")
        if st.button("Account Settings"):
            st.session_state.page = "account"
            st.rerun()
        
        st.write("---")
        # サイドバー: 信頼度閾値スライダー
        confidence = st.sidebar.slider(
            "Detection Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )

    # ファイルアップローダー
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # アップロードされたファイルの内容を一度だけ読み込む
        file_bytes = uploaded_file.getvalue() # getvalue() を使う方が一般的

        st.write("Processing...")
        api_error = None
        result_data = None

        # --- /detect/ API呼び出し ---
        try:
            files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
            params = {"confidence": confidence}
            detect_url = f"{API_URL}/detect/"
            logger.info(f"Sending request to {detect_url} with confidence {confidence}")

            response = requests.post(detect_url, files=files, params=params, headers=HEADERS)
            response.raise_for_status() # ステータスコードが 2xx でない場合に例外を発生させる

            result_data = response.json()
            logger.info("Received successful response from /detect/ API.")

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            api_error = f"API request failed: {e}"
            # サーバーからの詳細なエラーメッセージを表示しようと試みる
            try:
                error_detail = response.json().get("detail", str(e))
                api_error = f"API Error ({response.status_code}): {error_detail}"
            except: # JSONデコード失敗など
                pass
        except Exception as e:
             logger.error(f"An unexpected error occurred during API call: {e}", exc_info=True)
             api_error = f"An unexpected error occurred: {str(e)}"


        # --- 結果表示 ---
        if api_error:
            st.error(api_error)
        elif result_data and "error" in result_data:
             st.error(f"API returned an error: {result_data['error']}")
        elif result_data:
            # 入力画像と出力画像を横並びに表示
            col1, col2 = st.columns(2)
            with col1:
                try:
                    nparr = np.frombuffer(file_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, caption="Input Image", use_column_width=True)
                    else:
                        st.warning("Could not display input image.")
                except Exception as e:
                    st.warning(f"Error displaying input image: {e}")

            with col2:
                annotated_b64 = result_data.get("annotated_image", "")
                if annotated_b64:
                    try:
                        decoded_bytes = base64.b64decode(annotated_b64)
                        nparr = np.frombuffer(decoded_bytes, np.uint8)
                        annotated_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if annotated_img is not None:
                            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                            st.image(annotated_img_rgb, caption="Detection Results", use_column_width=True)
                        else:
                            st.error("Failed to decode annotated image from base64.")
                    except Exception as e:
                        st.error(f"Error displaying annotated image: {e}")
                else:
                    st.warning("No annotated image received from API.")

            # 検出結果の表示とフィードバックUI
            detections = result_data.get("detections", [])
            if detections:
                st.success(f"Detection completed! Processing time: {result_data.get('inference_time', 0):.2f} seconds")
                st.write(f"Objects detected: {len(detections)}")

                # セッションステートにフィードバックデータを初期化/保存
                if 'feedback_data' not in st.session_state or st.session_state.get('processed_image_name') != uploaded_file.name:
                    st.session_state.feedback_data = {
                        "image_base64": base64.b64encode(file_bytes).decode(), # 元画像をbase64で保存 (FeedbackDataモデルに合わせる)
                        "corrected_labels": [], # ここで初期化
                        "user_comment": "" # オプション
                    }
                    st.session_state.processed_image_name = uploaded_file.name

                current_feedback_labels = []
                st.write("### Detection Feedback")
                for i, detection in enumerate(detections):
                    # 一意なキーを生成
                    radio_key = f"feedback_radio_{uploaded_file.name}_{i}"
                    # セッションステートから前回の選択を取得（なければデフォルト）
                    # ここは改善の余地あり。毎回デフォルトに戻る可能性がある。
                    # 検出結果が変わるとインデックスも変わるため、より安定したキー管理が必要。
                    current_selection = "Undecided" # デフォルト

                    feedback = st.radio(
                        f"Detection #{i+1}: **{detection['class_name']}** (Conf: {detection['confidence']:.2%}) - Accurate?",
                        options=["Undecided", "Accurate", "Inaccurate"],
                        key=radio_key, # 一意なキー
                        index=0, # デフォルトは "Undecided"
                        horizontal=True
                    )
                    # ここでユーザーの選択を保存する（例：corrected_labelsに追加）
                    # この例では単純化のため、送信時に最新の選択を収集する
                    detection_feedback = {
                        "original_detection": detection, # 元の検出結果
                        "user_feedback": feedback, # ユーザーの評価
                        # "corrected_class": None # 不正確な場合に修正するUIをここに追加可能
                    }
                    current_feedback_labels.append(detection_feedback)

                    st.markdown("---")

                # セッションステートを更新
                st.session_state.feedback_data["corrected_labels"] = current_feedback_labels

                # フィードバック送信ボタン
                if st.button("Save Feedback"):
                    feedback_payload = st.session_state.feedback_data
                    feedback_url = f"{API_URL}/feedback/"
                    logger.info(f"Sending feedback data to {feedback_url}")
                    try:
                        # FeedbackData モデルに合わせてペイロードを調整する必要がある
                        # ここでは仮にそのまま送る
                        response = requests.post(feedback_url, json=feedback_payload, headers=HEADERS)
                        response.raise_for_status()

                        feedback_result = response.json()
                        if feedback_result.get("status") == "success":
                            st.success("Feedback saved successfully!")
                            logger.info("Feedback saved successfully via API.")
                            st.session_state.page = "recipe"
                            st.rerun()
                            # 送信成功したらセッションステートをクリアしても良い
                            # del st.session_state.feedback_data
                            # del st.session_state.processed_image_name
                        else:
                            st.error(f"Failed to save feedback: {feedback_result.get('detail', 'Unknown error')}")
                            logger.error(f"API returned error on feedback save: {feedback_result}")

                    except requests.exceptions.RequestException as e:
                        logger.error(f"Feedback API request failed: {e}")
                        api_error = f"Feedback API request failed: {e}"
                        try:
                            error_detail = response.json().get("detail", str(e))
                            api_error = f"API Error ({response.status_code}): {error_detail}"
                        except: pass
                        st.error(api_error)
                    except Exception as e:
                        logger.error(f"An unexpected error occurred during feedback submission: {e}", exc_info=True)
                        st.error(f"An unexpected error occurred: {str(e)}")

            else: # if detections:
                st.info("No objects detected with the current confidence threshold.")

        # else: result_data is None or contains error handled above

    else: # if uploaded_file is not None:
        st.info("👆 Please upload an image file to start detection.")
def recipe_suggestion_page():
    """ページのレシピ提案（フランス語）"""
    st.title("🍳 Suggestions de Recettes IA")
    
    if "feedback_data" not in st.session_state:
        st.error("Aucune donnée de détection disponible. Veuillez d'abord détecter les objets.")
        if st.button("Retour à la Détection"):
            st.session_state.page = "main"
            st.rerun()
        return

    # 検出されたオブジェクトからレシピのプロンプトを生成
    detected_items = []
    for label in st.session_state.feedback_data["corrected_labels"]:
        if label["user_feedback"] == "Accurate":
            detected_items.append(label["original_detection"]["class_name"])
    
    if not detected_items:
        st.warning("Aucune détection confirmée pour générer des recettes.")
        if st.button("Retour à la Détection"):
            st.session_state.page = "main"
            st.rerun()
        return

    # Ollamaへのプロンプト生成（フランス語）
    ingredients_list = ", ".join(detected_items)
    prompt = f"""Ingrédients détectés: {ingredients_list}

    Proposez 3 recettes simples utilisant ces ingrédients.
    Pour chaque recette, incluez:
    - Nom de la recette
    - Temps de préparation
    - Ingrédients supplémentaires nécessaires et ces prix
    - Instructions (environ 5 étapes)

    Répondez en français."""

    try:
        # Ollamaへのリクエスト
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            recipe_response = response.json()
            recipes_text = recipe_response["response"]
            
            # レシピの表示（フランス語）
            st.write("### 🥘 Ingrédients Détectés")
            st.write(f"**{ingredients_list}**")
            
            st.write("### 📝 Recettes Recommandées")
            st.markdown(recipes_text)
            
            # 新しい検出を行うためのボタン
            if st.button("Nouvelle Détection"):
                st.session_state.page = "main"
                if "feedback_data" in st.session_state:
                    del st.session_state.feedback_data
                st.rerun()
                
        else:
            st.error("Échec de la génération des recettes.")
            
    except Exception as e:
        st.error(f"Erreur lors de la génération des recettes: {str(e)}")
        logger.error(f"Erreur de génération de recette: {e}", exc_info=True)

# --- メインルーティング ---
def main():
    """アプリケーションのメインエントリーポイントとページルーティング"""
    # セッション状態の初期化
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "username" not in st.session_state:
        st.session_state.username = None

    # ページルーティング
    if not st.session_state.logged_in:
        if st.session_state.page == "login":
            login_page()
        elif st.session_state.page == "register":
            register_page()
        else: # デフォルトはログインページ
             login_page()
    else:
        if st.session_state.page == "account":
            user_account_page()
        elif st.session_state.page == "recipe":
            recipe_suggestion_page()
        else:
            main_app()

if __name__ == "__main__":
    main()

