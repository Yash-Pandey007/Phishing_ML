import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import time
import warnings
import ipaddress
import tldextract # Needed for the new extraction function
from urllib.parse import urlparse, parse_qs

# --- Configuration ---
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning)

# --- File Paths ---
MODEL_PATH = "best_model.pkl"
LE_SCHEME_PATH = "le_scheme.pkl"
LE_NETLOC_PATH = "le_netloc.pkl"
LE_PATH_PATH = "le_path.pkl"
PT_PATH = "power_transformer.pkl"

# --->>> THIS MUST BE THE FIRST STREAMLIT COMMAND <<<---
st.set_page_config(
    page_title="Collaborative Phishing Detector",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Load Model and Preprocessors ---
@st.cache_resource
def load_objects():
    """Loads the model and preprocessor objects from pickle files."""
    model, le_scheme, le_netloc, le_path, pt, expected_feature_order = None, None, None, None, None, None # Initialize to None
    try:
        with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)
        with open(LE_SCHEME_PATH, 'rb') as f: le_scheme = pickle.load(f)
        with open(LE_NETLOC_PATH, 'rb') as f: le_netloc = pickle.load(f)
        with open(LE_PATH_PATH, 'rb') as f: le_path = pickle.load(f)
        with open(PT_PATH, 'rb') as f: pt = pickle.load(f)
        try:
            expected_feature_order = pt.feature_names_in_
            print("Features expected by PowerTransformer:", expected_feature_order)
        except AttributeError:
            st.error("🚨 PowerTransformer object loaded does not have 'feature_names_in_'. Cannot determine expected feature order automatically.")
            expected_feature_order = None
        except Exception as e_feat:
             st.error(f"🚨 Error getting feature names from PowerTransformer: {e_feat}")
             expected_feature_order = None
        return model, le_scheme, le_netloc, le_path, pt, expected_feature_order
    except FileNotFoundError as e:
        st.error(f"🚨 Error loading required file: {e}. Ensure all .pkl files are in the script's directory.")
        return model, le_scheme, le_netloc, le_path, pt, expected_feature_order
    except Exception as e:
        st.error(f"🚨 An unexpected error occurred loading objects: {e}")
        return model, le_scheme, le_netloc, le_path, pt, expected_feature_order

svc_model, le_scheme, le_netloc, le_path, pt, EXPECTED_FEATURE_ORDER = load_objects()

# --- Combined Feature Extraction Function (FOR MODEL INPUT) ---
def extract_all_features_for_model(url):
    """
    Extracts ALL features expected by the trained model/preprocessor.
    Returns a dictionary of features or None if expected order is unknown.
    """
    features = {}
    # --- CORRECTED CHECK ---
    if EXPECTED_FEATURE_ORDER is not None: # Check if the list/array itself is not None
        for feature_name in EXPECTED_FEATURE_ORDER:
             if any(sub in feature_name for sub in ['ratio', 'avg']): features[feature_name] = 0.0
             elif any(sub in feature_name for sub in ['age', 'length', 'rank', 'traffic', 'redirection']): features[feature_name] = -1
             else: features[feature_name] = 0
    else:
        st.error("🚨 Cannot extract features: Expected feature order is unknown.")
        return None

    if not re.match(r'^[a-zA-Z]+://', url): url = 'http://' + url

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname if parsed.hostname else ''
        path = parsed.path if parsed.path else ''
        query = parsed.query if parsed.query else ''
        full_url_text = hostname + path + query

        ext = tldextract.extract(url); domain = ext.domain; subdomain = ext.subdomain; suffix = ext.suffix

        # --- Calculate features and update the 'features' dictionary ---
        # (Ensure keys are checked against EXPECTED_FEATURE_ORDER if adding/removing features)
        features['length_url'] = len(url)
        if 'url_length' in features: features['url_length'] = len(url)
        features['length_hostname'] = len(hostname)
        features['nb_dots'] = url.count('.')
        features['nb_hyphens'] = url.count('-')
        if 'nb_and' in features: features['nb_and'] = url.count('&')
        if 'nb_underscore' in features: features['nb_underscore'] = url.count('_')
        if 'nb_percent' in features: features['nb_percent'] = url.count('%')
        if 'nb_slash' in features: features['nb_slash'] = url.count('/')
        if 'nb_semicolumn' in features: features['nb_semicolumn'] = url.count(';')
        if 'nb_space' in features: features['nb_space'] = url.count(' ') + url.count('%20')
        if 'nb_at' in features: features['nb_at'] = url.count('@')
        if 'nb_eq' in features: features['nb_eq'] = url.count('=')
        if 'nb_tilde' in features: features['nb_tilde'] = url.count('~')
        if 'nb_star' in features: features['nb_star'] = url.count('*')
        if 'nb_colon' in features: features['nb_colon'] = url.count(':')
        if 'nb_comma' in features: features['nb_comma'] = url.count(',')
        if 'nb_dollar' in features: features['nb_dollar'] = url.count('$')
        if 'nb_dslash' in features: features['nb_dslash'] = url.count('//')
        if 'http_in_path' in features: features['http_in_path'] = 1 if 'http' in path.lower() else 0

        if 'nb_www' in features: features['nb_www'] = 1 if 'www' in subdomain.split('.') else 0
        if 'nb_com' in features: features['nb_com'] = 1 if suffix == 'com' else 0
        if 'nb_subdomains' in features: features['nb_subdomains'] = len(subdomain.split('.')) if subdomain else 0
        if 'prefix_suffix' in features: features['prefix_suffix'] = 1 if domain.startswith('-') or domain.endswith('-') else 0
        if 'tld_in_path' in features: features['tld_in_path'] = 1 if suffix and suffix in path else 0
        if 'tld_in_subdomain' in features: features['tld_in_subdomain'] = 1 if suffix and suffix in subdomain else 0
        if 'abnormal_subdomain' in features: features['abnormal_subdomain'] = 1 if suffix and subdomain.endswith(f'.{suffix}') else 0

        features['scheme'] = parsed.scheme if parsed.scheme else 'unknown'
        if 'https_token' in features: features['https_token'] = 1 if 'https' in url.lower() else 0

        features['netloc'] = parsed.netloc if parsed.netloc else 'unknown'
        features['path'] = path
        features['num_path_segments'] = len([s for s in path.split('/') if s])
        try: features['num_query_params'] = len(parse_qs(query))
        except ValueError: features['num_query_params'] = url.count('&') + 1 if '?' in url and query else 0

        try: ipaddress.ip_address(hostname); features['is_ip'] = 1
        except ValueError: features['is_ip'] = 0
        if 'ip' in features: features['ip'] = features['is_ip']

        words = re.split(r'\W+', full_url_text); words = [w for w in words if w]
        word_lengths = [len(w) for w in words] if words else [0]
        host_words = re.split(r'\W+', hostname); host_words = [w for w in host_words if w]
        host_word_lengths = [len(w) for w in host_words] if host_words else [0]
        path_words = re.split(r'\W+', path); path_words = [w for w in path_words if w]
        path_word_lengths = [len(w) for w in path_words] if path_words else [0]

        if 'length_words_raw' in features: features['length_words_raw'] = len(words)
        if 'shortest_words_raw' in features: features['shortest_words_raw'] = min(word_lengths)
        if 'longest_words_raw' in features: features['longest_words_raw'] = max(word_lengths)
        if 'avg_words_raw' in features: features['avg_words_raw'] = np.mean(word_lengths) if words else 0
        if 'shortest_word_host' in features: features['shortest_word_host'] = min(host_word_lengths)
        if 'longest_word_host' in features: features['longest_word_host'] = max(host_word_lengths)
        if 'avg_word_host' in features: features['avg_word_host'] = np.mean(host_word_lengths) if host_words else 0
        if 'shortest_word_path' in features: features['shortest_word_path'] = min(path_word_lengths)
        if 'longest_word_path' in features: features['longest_word_path'] = max(path_word_lengths)
        if 'avg_word_path' in features: features['avg_word_path'] = np.mean(path_word_lengths) if path_words else 0

        if 'char_repeat' in features: features['char_repeat'] = sum(1 for i in range(len(full_url_text) - 1) if full_url_text[i] == full_url_text[i+1])

        digits_url = sum(c.isdigit() for c in url)
        digits_host = sum(c.isdigit() for c in hostname)
        if 'ratio_digits_url' in features: features['ratio_digits_url'] = digits_url / features['length_url'] if features['length_url'] > 0 else 0
        if 'ratio_digits_host' in features: features['ratio_digits_host'] = digits_host / features['length_hostname'] if features['length_hostname'] > 0 else 0

        # Placeholders remain as initialized (-1 or 0)

    except Exception as e:
        st.warning(f"⚠️ Error during feature extraction: {e}. Using default values for some features.")

    # --- CORRECTED CHECK (Final Check) ---
    final_features = {}
    if EXPECTED_FEATURE_ORDER is not None: # Check if the list/array itself is not None
        for key in EXPECTED_FEATURE_ORDER:
            final_features[key] = features.get(key, 0 if not any(sub in key for sub in ['ratio','avg','age','length','rank','traffic','redirection']) else (-1 if any(sub in key for sub in ['age','length','rank','traffic','redirection']) else 0.0))
    else:
        st.error("🚨 Cannot finalize features: Expected feature order is unknown.")
        return None
    return final_features


# --- User Input Section (Modified for Clarity) ---
def get_user_inputs(expected_features):
    """Collect features that cannot be automatically extracted."""
    # ... (Function definition remains the same) ...
    st.subheader("Step 2: Provide Additional Context (Optional)")
    st.markdown("This information helps assess the site but **cannot** be used by the current prediction model, which relies only on URL structure.")
    user_data = {}
    default_value = -1
    user_input_keys = [
        'nb_hyperlinks', 'ratio_intHyperlinks', 'nb_extCSS', 'external_favicon',
        'ratio_intMedia', 'ratio_extMedia', 'safe_anchor', 'domain_in_title',
        'domain_with_copyright', 'domain_registration_length', 'domain_age',
        'web_traffic', 'google_index', 'page_rank'
    ]

    with st.form("user_features_form"):
        st.markdown("**Page Content & Links**")
        nb_hyperlinks = st.number_input("Approx. Number of links on page:", min_value=0, value=10, key="links")
        ratio_intHyperlinks = st.slider("Approx. Percentage of internal links (%):", 0, 100, 50, key="intlinks") / 100.0
        safe_anchor = st.checkbox("Do most links seem to match their destination (safe anchors)?", value=True, key="safeanchor")

        st.markdown("**External Resources & Security**")
        nb_extCSS = st.number_input("Approx. Number of external CSS files:", min_value=0, value=1, key="css")
        external_favicon = st.checkbox("Does the site use a favicon hosted on a different domain?", value=False, key="favicon")
        domain_in_title = st.checkbox("Is the main domain name present in the browser page title?", value=True, key="title")
        domain_with_copyright = st.checkbox("Is the main domain name mentioned near a copyright notice?", value=False, key="copyright")

        st.markdown("**Media Content**")
        ratio_intMedia = st.slider("Approx. Percentage of images/media hosted internally (%):", 0, 100, 80, key="intmedia") / 100.0
        ratio_extMedia = 1.0 - ratio_intMedia

        st.markdown("**Domain Registration & Reputation**")
        col1, col2 = st.columns(2)
        with col1:
            domain_age_years = st.number_input("Approx. Domain age (in years):", min_value=0, max_value=30, value=1, key="age")
            domain_age = domain_age_years * 365
        with col2:
            domain_registration_length = st.selectbox("Domain registration period (years):", [1, 2, 3, 5, 10, default_value], index=0, key="reglen", format_func=lambda x: f"{x} years" if x != default_value else "Unknown")

        web_traffic_cat = st.select_slider("Estimated Web Traffic/Popularity:", options=['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Unknown'], value='Medium', key="traffic")
        traffic_map = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5, 'Unknown': default_value}
        web_traffic = traffic_map[web_traffic_cat]

        google_index = st.checkbox("Is the site likely indexed by Google?", value=True, key="gindex")
        page_rank = st.slider("Estimated PageRank / Authority (0-10):", 0, 10, 5, key="pagerank")

        submitted = st.form_submit_button("Submit Context & Get Prediction")

        if submitted:
            user_data = {
                'nb_hyperlinks': nb_hyperlinks, 'ratio_intHyperlinks': ratio_intHyperlinks,
                'safe_anchor': int(safe_anchor), 'nb_extCSS': nb_extCSS,
                'external_favicon': int(external_favicon), 'domain_in_title': int(domain_in_title),
                'domain_with_copyright': int(domain_with_copyright), 'ratio_intMedia': ratio_intMedia,
                'ratio_extMedia': ratio_extMedia, 'domain_age': domain_age,
                'domain_registration_length': domain_registration_length, 'web_traffic': web_traffic,
                'google_index': int(google_index), 'page_rank': page_rank
            }
            # Add default values for any expected features the user didn't provide, if necessary
            # for key in expected_features:
            #     if key not in user_data and key in user_input_keys: # Check if it's a key we expected user input for
            #          user_data[key] = default_value # Or another appropriate default
            return user_data
    return None


# --- Preprocessing Function (FOR MODEL INPUT) ---
def preprocess_features_for_model(features_dict, le_scheme, le_netloc, le_path, pt, expected_feature_order):
    """Applies loaded LabelEncoders and PowerTransformer to the full feature set."""
    # ... (Function definition remains the same) ...
    if not all([le_scheme, le_netloc, le_path, pt]):
         st.error("🚨 Preprocessing objects not loaded correctly.")
         return None
    # --- CORRECTED CHECK ---
    if expected_feature_order is None: # Check if None explicitly
         st.error("🚨 Cannot preprocess: Expected feature order is unknown.")
         return None
    if features_dict is None:
         st.error("🚨 Cannot preprocess: Feature extraction failed.")
         return None

    df = pd.DataFrame([features_dict])

    try:
        if 'scheme' in df.columns: df['scheme'] = le_scheme.transform(df['scheme'])
        else: st.warning("⚠️ 'scheme' column not found for LabelEncoding.")
    except ValueError: st.warning(f"⚠️ Unseen category in 'scheme': {df['scheme'].iloc[0]}. Assigning default value (-1)."); df['scheme'] = -1
    except KeyError: st.warning("⚠️ 'scheme' column missing, cannot apply LabelEncoder.")

    try:
        if 'netloc' in df.columns: df['netloc'] = le_netloc.transform(df['netloc'])
        else: st.warning("⚠️ 'netloc' column not found for LabelEncoding.")
    except ValueError: unseen_netloc = df['netloc'].iloc[0]; st.warning(f"⚠️ Unseen category in 'netloc': {unseen_netloc[:100]}{'...' if len(unseen_netloc)>100 else ''}. Assigning default value (-1)."); df['netloc'] = -1
    except KeyError: st.warning("⚠️ 'netloc' column missing, cannot apply LabelEncoder.")

    try:
        if 'path' in df.columns: df['path'] = le_path.transform(df['path'])
        else: st.warning("⚠️ 'path' column not found for LabelEncoding.")
    except ValueError: unseen_path = df['path'].iloc[0]; st.warning(f"⚠️ Unseen category in 'path': {unseen_path[:100]}{'...' if len(unseen_path)>100 else ''}. Assigning default value (-1)."); df['path'] = -1
    except KeyError: st.warning("⚠️ 'path' column missing, cannot apply LabelEncoder.")

    try:
        for col in expected_feature_order:
            if col not in df.columns:
                st.warning(f"Column '{col}' was missing before PowerTransform. Filling with 0.")
                df[col] = 0
        df_ordered = df[expected_feature_order]
        transformed_features = pt.transform(df_ordered)
        return transformed_features
    except KeyError as e: st.error(f"🚨 Feature mismatch error during preprocessing: Missing column {e}. Expected order: {expected_feature_order}."); return None
    except ValueError as e: st.error(f"🚨 Error during Power Transformation: {e}. Check input features for invalid values (NaN, infinity) or issues caused by unseen label placeholders (-1)."); return None
    except Exception as e: st.error(f"🚨 An unexpected error occurred during power transformation: {e}"); return None


# --- Streamlit App User Interface ---
st.title("🎣 Phishing URL Detector")
st.markdown("""
Analyzes URL structure and uses a pre-trained model for prediction.
Optionally provide context about the site (content, reputation) for a more informed assessment (Note: user context **not** used by the current prediction model).
""")
st.markdown("---")

st.subheader("Step 1: Enter the URL to Analyze")
url_input = st.text_input(
    "URL:",
    placeholder="e.g., http://www.google.com or http://example-phishing-site.com/login.html",
    key="url_input_field"
)

# --- CORRECTED CHECK ---
if url_input and all(obj is not None for obj in [svc_model, le_scheme, le_netloc, le_path, pt, EXPECTED_FEATURE_ORDER]):
    model_input_features = extract_all_features_for_model(url_input)

    if model_input_features is not None:
        with st.expander("View Automated URL Structure Analysis Results"):
             display_subset = {k: v for k, v in model_input_features.items() if k in ['length_url', 'length_hostname', 'nb_dots', 'nb_hyphens', 'nb_subdomains', 'https_token', 'num_path_segments', 'num_query_params', 'is_ip', 'scheme', 'netloc', 'path']}
             st.json(display_subset)

        user_provided_context = get_user_inputs(EXPECTED_FEATURE_ORDER)

        if user_provided_context is not None:
            st.markdown("---")
            st.subheader("Step 3: Analysis & Prediction")

            with st.expander("View User-Provided Context"):
                 st.json(user_provided_context)

            with st.spinner("⚙️ Preprocessing features for model..."):
                preprocessed_data_for_model = preprocess_features_for_model(
                    model_input_features, le_scheme, le_netloc, le_path, pt, EXPECTED_FEATURE_ORDER
                )

            if preprocessed_data_for_model is not None:
                 with st.spinner("🤖 Predicting using the trained model..."):
                    prediction_start_time = time.time()
                    try:
                        prediction = svc_model.predict(preprocessed_data_for_model)
                        prediction_label = prediction[0]
                        prediction_end_time = time.time()
                        st.info(f"Prediction completed in {prediction_end_time - prediction_start_time:.2f} seconds.")

                        st.markdown("**Prediction Result (Based on URL Structure Model):**")
                        if prediction_label == 'phishing':
                            st.error("🚨 Prediction: **Phishing**")
                            st.image("https://cdn-icons-png.flaticon.com/512/10180/10180609.png", caption="Warning: Likely Phishing", width=80)
                        elif prediction_label == 'legitimate':
                            st.success("✅ Prediction: **Legitimate**")
                            st.image("https://cdn-icons-png.flaticon.com/512/1828/1828640.png", caption="Verified: Likely Legitimate", width=80)
                        else:
                            st.warning(f"❓ Unknown prediction result: {prediction_label}")
                        st.caption("Note: The prediction uses the model trained on URL structural features. Your provided context adds valuable human insight but was not used in this specific model's calculation.")

                    except Exception as e:
                        st.error(f"🚨 An error occurred during model prediction: {e}")
            else:
                st.error("🚨 Could not preprocess features required by the model. Prediction aborted.")
    else:
         st.error("🚨 Feature extraction failed. Cannot proceed.")

# --- CORRECTED CHECK ---
elif not all(obj is not None for obj in [svc_model, le_scheme, le_netloc, le_path, pt, EXPECTED_FEATURE_ORDER]):
    st.error("🚨 Application cannot start because essential model/preprocessor files failed to load or feature order could not be determined. Please check file paths and console logs.")

st.markdown("---")
st.caption("Powered by Streamlit | Model: SVC | Analyzes URL structure.")
