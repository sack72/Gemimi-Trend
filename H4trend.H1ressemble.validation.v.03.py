import pandas as pd
import numpy as np
import io
from google.colab import files

# ==========================================
# 1. TIME FRAME MANAGEMENT (H1 -> H4)
# ==========================================

def convert_h1_to_h4(df_h1):
    """
    Resamples H1 Data to H4 Data to ensure perfect synchronization.
    """
    print("   ... Resampling H1 data to H4 ...")
    df = df_h1.copy()
    
    # Ensure datetime
    try:
        df['Local time'] = pd.to_datetime(df['Local time'], dayfirst=True)
    except:
        df['Local time'] = pd.to_datetime(df['Local time'])
        
    df = df.set_index('Local time')
    
    # Resample Logic (4-Hour Bars)
    df_h4 = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    
    df_h4 = df_h4.reset_index()
    return df_h4

# ==========================================
# 2. CORE STRATEGY LOGIC (MACRO ONLY)
# ==========================================

def identify_swings_and_fvgs(df):
    """
    Identifies Swing Highs/Lows with OC Confirmation and Fair Value Gaps (FVG).
    """
    df = df.copy()
    # Initialize columns
    df['Swing_Low'] = False
    df['Swing_High'] = False
    df['Swing_Low_OC_Idx'] = np.nan
    df['Swing_High_OC_Idx'] = np.nan
    df['Bearish_FVG'] = False
    df['Bullish_FVG'] = False

    # --- SWINGS ---
    for i in range(1, len(df) - 1):
        # Swing Low
        if df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1]:
            is_bullish = df['Close'][i] > df['Open'][i]
            if is_bullish:
                df.at[i, 'Swing_Low'] = True; df.at[i, 'Swing_Low_OC_Idx'] = i
            else:
                for j in range(i + 1, len(df)):
                    if df['Low'][j] < df['Low'][i]: break
                    if df['Close'][j] > df['Open'][j] and df['Low'][j] >= df['Low'][i]:
                        df.at[i, 'Swing_Low'] = True; df.at[i, 'Swing_Low_OC_Idx'] = j; break

        # Swing High
        if df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1]:
            is_bearish = df['Close'][i] < df['Open'][i]
            if is_bearish:
                df.at[i, 'Swing_High'] = True; df.at[i, 'Swing_High_OC_Idx'] = i
            else:
                for j in range(i + 1, len(df)):
                    if df['High'][j] > df['High'][i]: break
                    if df['Close'][j] < df['Open'][j] and df['High'][j] <= df['High'][i]:
                        df.at[i, 'Swing_High'] = True; df.at[i, 'Swing_High_OC_Idx'] = j; break

    # --- FVGs ---
    for i in range(len(df) - 2):
        if df['Low'][i] > df['High'][i+2]: df.at[i+2, 'Bearish_FVG'] = True
        if df['High'][i] < df['Low'][i+2]: df.at[i+2, 'Bullish_FVG'] = True

    return df

def find_macro_legs(df):
    legs = []

    # --- SAFE VALIDATION HELPER ---
    # The 'Validation_Time' is the EARLIEST moment we knew this was a valid leg.
    # It requires 2 conditions:
    # 1. The OC Candle has closed (Color confirmation).
    # 2. The (i+1) Candle has closed (Geometric confirmation of the V-shape).
    def get_safe_validation_time(swing_idx, oc_idx):
        try:
            time_oc = df.at[int(oc_idx), 'Local time']
            
            # We strictly need the (i+1) candle to exist to define the swing geometry
            if swing_idx + 1 < len(df):
                time_next_candle = df.at[swing_idx + 1, 'Local time']
                # The validation is the LATER of the two events
                return max(time_oc, time_next_candle)
            
            # Fallback if at end of data
            return time_oc
        except:
            return None

    # --- 1. INITIALIZATION ---
    first_leg = None
    first_leg_end_idx = float('inf')

    # A) Check for First Valid DOWN Leg
    swing_lows = df.index[df['Swing_Low'] == True].tolist()
    for low_idx in swing_lows:
        if low_idx >= first_leg_end_idx: break
        oc_idx = int(df.at[low_idx, 'Swing_Low_OC_Idx'])

        fvg_idx = -1
        for k in range(oc_idx, -1, -1):
            if df.at[k, 'Bearish_FVG']: fvg_idx = k; break

        if fvg_idx != -1:
            high_idx = -1
            for m in range(fvg_idx, -1, -1):
                if df.at[m, 'Swing_High']: high_idx = m; break

            if high_idx != -1:
                if low_idx < first_leg_end_idx:
                    first_leg = {
                        'id': 1, 'Type': 'DOWN',
                        'Start_Time': df.at[high_idx, 'Local time'], 'Start_Price': df.at[high_idx, 'High'],
                        'End_Time': df.at[low_idx, 'Local time'], 'End_Price': df.at[low_idx, 'Low'],
                        'Validation_Time': get_safe_validation_time(low_idx, oc_idx),
                        'Start_Idx': high_idx, 'End_Idx': low_idx
                    }
                    first_leg_end_idx = low_idx
                    break

    # B) Check for First Valid UP Leg
    swing_highs = df.index[df['Swing_High'] == True].tolist()
    for high_idx in swing_highs:
        if high_idx >= first_leg_end_idx: break
        oc_idx = int(df.at[high_idx, 'Swing_High_OC_Idx'])

        fvg_idx = -1
        for k in range(oc_idx, -1, -1):
            if df.at[k, 'Bullish_FVG']: fvg_idx = k; break

        if fvg_idx != -1:
            low_idx = -1
            for m in range(fvg_idx, -1, -1):
                if df.at[m, 'Swing_Low']: low_idx = m; break

            if low_idx != -1:
                if high_idx < first_leg_end_idx:
                    first_leg = {
                        'id': 1, 'Type': 'UP',
                        'Start_Time': df.at[low_idx, 'Local time'], 'Start_Price': df.at[low_idx, 'Low'],
                        'End_Time': df.at[high_idx, 'Local time'], 'End_Price': df.at[high_idx, 'High'],
                        'Validation_Time': get_safe_validation_time(high_idx, oc_idx),
                        'Start_Idx': low_idx, 'End_Idx': high_idx
                    }
                    first_leg_end_idx = high_idx
                    break

    if not first_leg: return pd.DataFrame()
    legs.append(first_leg)

    # --- 2. SEQUENTIAL MONITORING ---
    search_idx = legs[-1]['End_Idx']

    while search_idx < len(df) - 1:
        current_leg = legs[-1]
        leg_type = current_leg['Type']

        continuation_trigger = current_leg['End_Price'] # Low for DOWN, High for UP
        reversal_trigger = current_leg['Start_Price']   # High for DOWN, Low for UP

        event_found = False

        for i in range(search_idx + 1, len(df)):
            if event_found: break

            curr_low = df.at[i, 'Low']
            curr_high = df.at[i, 'High']

            # --- A. CHECK CONTINUATION ---
            is_continuation = (leg_type == 'DOWN' and curr_low < continuation_trigger) or \
                              (leg_type == 'UP' and curr_high > continuation_trigger)

            if is_continuation:
                if leg_type == 'DOWN':
                    next_swings = df.index[(df['Swing_Low'] == True) & (df.index >= i)].tolist()
                else:
                    next_swings = df.index[(df['Swing_High'] == True) & (df.index >= i)].tolist()

                if not next_swings:
                    search_idx = len(df); event_found = True; break

                swing_idx = next_swings[0]
                oc_idx = int(df.at[swing_idx, 'Swing_Low_OC_Idx']) if leg_type == 'DOWN' else int(df.at[swing_idx, 'Swing_High_OC_Idx'])
                prev_leg_end_idx = current_leg['End_Idx']

                # Check FVG
                has_fvg = False
                for k in range(oc_idx, prev_leg_end_idx, -1):
                    if leg_type == 'DOWN' and df.at[k, 'Bearish_FVG']: has_fvg = True; break
                    if leg_type == 'UP' and df.at[k, 'Bullish_FVG']: has_fvg = True; break

                if has_fvg:
                    # NEW LEG CREATED
                    fvg_idx = k
                    start_point_idx = -1
                    for m in range(fvg_idx, -1, -1):
                        if leg_type == 'DOWN' and df.at[m, 'Swing_High']: start_point_idx = m; break
                        if leg_type == 'UP' and df.at[m, 'Swing_Low']: start_point_idx = m; break

                    if start_point_idx != -1:
                        # --- MERGE LOGIC FIX ---
                        if start_point_idx == current_leg['Start_Idx']:
                             # SAME ANCHOR -> UPDATE EXISTING LEG (Extension)
                             legs[-1]['End_Time'] = df.at[swing_idx, 'Local time']
                             legs[-1]['End_Price'] = df.at[swing_idx, 'Low' if leg_type=='DOWN' else 'High']
                             legs[-1]['Validation_Time'] = get_safe_validation_time(swing_idx, oc_idx) # UPDATE VALIDATION
                             legs[-1]['End_Idx'] = swing_idx
                        else:
                             # NEW ANCHOR -> CREATE NEW LEG
                             legs.append({
                                'id': len(legs) + 1, 'Type': leg_type,
                                'Start_Time': df.at[start_point_idx, 'Local time'], 'Start_Price': df.at[start_point_idx, 'High' if leg_type=='DOWN' else 'Low'],
                                'End_Time': df.at[swing_idx, 'Local time'], 'End_Price': df.at[swing_idx, 'Low' if leg_type=='DOWN' else 'High'],
                                'Validation_Time': get_safe_validation_time(swing_idx, oc_idx),
                                'Start_Idx': start_point_idx, 'End_Idx': swing_idx
                             })
                        search_idx = swing_idx; event_found = True
                else:
                    # EXTENSION (No FVG) -> Update in place
                    legs[-1]['End_Time'] = df.at[swing_idx, 'Local time']
                    legs[-1]['End_Price'] = df.at[swing_idx, 'Low' if leg_type=='DOWN' else 'High']
                    legs[-1]['Validation_Time'] = get_safe_validation_time(swing_idx, oc_idx) # UPDATE VALIDATION
                    legs[-1]['End_Idx'] = swing_idx
                    search_idx = swing_idx; event_found = True

            # --- B. CHECK REVERSAL ---
            elif (leg_type == 'DOWN' and curr_high > reversal_trigger) or \
                 (leg_type == 'UP' and curr_low < reversal_trigger):

                new_type = 'UP' if leg_type == 'DOWN' else 'DOWN'

                if new_type == 'UP':
                    next_swings = df.index[(df['Swing_High'] == True) & (df.index >= i)].tolist()
                else:
                    next_swings = df.index[(df['Swing_Low'] == True) & (df.index >= i)].tolist()

                if not next_swings:
                    search_idx = len(df); event_found = True; break

                swing_idx = next_swings[0]
                oc_idx = int(df.at[swing_idx, 'Swing_High_OC_Idx']) if new_type == 'UP' else int(df.at[swing_idx, 'Swing_Low_OC_Idx'])

                # Rule: Scan from OC back to Anchor
                anchor_idx = current_leg['End_Idx']

                # 1. Find Most Recent FVG (closest to OC)
                fvg_idx = -1
                for k in range(oc_idx, anchor_idx, -1):
                    if new_type == 'UP' and df.at[k, 'Bullish_FVG']: fvg_idx = k; break 
                    if new_type == 'DOWN' and df.at[k, 'Bearish_FVG']: fvg_idx = k; break 

                if fvg_idx != -1:
                    # 2. Find Most Recent Swing High/Low BEFORE the FVG
                    new_anchor_idx = -1
                    for m in range(fvg_idx, -1, -1):
                        if new_type == 'UP' and df.at[m, 'Swing_Low']: new_anchor_idx = m; break
                        if new_type == 'DOWN' and df.at[m, 'Swing_High']: new_anchor_idx = m; break

                    if new_anchor_idx != -1:
                        # VALID REVERSAL -> New Leg
                        legs.append({
                            'id': len(legs) + 1, 'Type': new_type,
                            'Start_Time': df.at[new_anchor_idx, 'Local time'], 'Start_Price': df.at[new_anchor_idx, 'Low' if new_type=='UP' else 'High'],
                            'End_Time': df.at[swing_idx, 'Local time'], 'End_Price': df.at[swing_idx, 'High' if new_type=='UP' else 'Low'],
                            'Validation_Time': get_safe_validation_time(swing_idx, oc_idx),
                            'Start_Idx': new_anchor_idx, 'End_Idx': swing_idx
                        })
                        search_idx = swing_idx; event_found = True

        if not event_found: break

    # 3. Clean Duplicates
    raw_df = pd.DataFrame(legs)
    cleaned_legs = []
    if not raw_df.empty:
        grouped = raw_df.groupby('id')
        for name, group in grouped:
            final_leg = group.iloc[-1]
            cleaned_legs.append(final_leg)

    cleaned_df = pd.DataFrame(cleaned_legs).reset_index(drop=True)
    return cleaned_df

# ==========================================
# 3. EXECUTION
# ==========================================

print("👇 Please upload your **H1 (ONE HOUR)** CSV file:")
uploaded = files.upload()

for fn in uploaded.keys():
    print(f"\n📂 PROCESSING FILE: {fn}")
    try:
        # Load H1
        df_h1 = pd.read_csv(io.BytesIO(uploaded[fn]))
        print(f"   ℹ️ H1 Data Loaded. Rows: {len(df_h1)}")

        # Resample to H4
        df_h4 = convert_h1_to_h4(df_h1)
        print(f"   ℹ️ H4 Data Generated. Rows: {len(df_h4)}")

        # Run MACRO Logic
        df_processed = identify_swings_and_fvgs(df_h4)
        legs_df = find_macro_legs(df_processed)

        if not legs_df.empty:
            output_filename = fn.replace(".csv", "_H4_MACRO_STRUCTURE.csv")
            
            # Select relevant columns for verification
            cols = ['id', 'Type', 'Start_Time', 'Start_Price', 'End_Time', 'End_Price', 'Validation_Time']
            legs_df = legs_df[cols]
            
            legs_df.to_csv(output_filename, index=False)
            print(f"✅ Success! Generated {output_filename}")
            print("\n👀 PREVIEW (Check 'Validation_Time' column):")
            print(legs_df.head(10).to_markdown(index=False))
            files.download(output_filename)
        else:
            print("❌ Analysis finished but NO Valid Legs were found.")

    except Exception as e:
        print(f"⚠️ Error: {e}")
