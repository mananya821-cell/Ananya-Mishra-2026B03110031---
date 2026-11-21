# Project 13: Hotel Booking & Occupancy Analysis
# Student Name: Ananya Mishra
# Student ID  : 2026B03110031
# Libraries   : ONLY numpy and pandas
# How to run  : open a terminal in this folder and type:  python hotel_analysis.py
# Input       : hotel_bookings_unique.csv
# Output      : cleaned_hotel_bookings.csv   (cleaned dataset)
#               results/project13_results_sheet.csv   (ALL results in ONE sheet)

import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "hotel_bookings_unique.csv"
CLEAN_FILE = PROJECT_DIR / "cleaned_hotel_bookings.csv"
SHEET_FILE = PROJECT_DIR / "results" / "project13_results_sheet.csv"

# ---------------------------------------------------------------------
# Helper functions and lambda functions
# ---------------------------------------------------------------------
results = []          # every table is stored here and written to ONE sheet at the end

def show(title, table, index=True):
    """Print a table on screen and store it for the single results sheet."""
    print(title)
    print(table.to_string(index=index))
    print()
    results.append((title, table, index))

def metrics(**values):
    """Turn name=value pairs into a small two-column table."""
    return pd.DataFrame({"Metric": list(values.keys()), "Value": list(values.values())})

def iqr_upper_fence(values):
    """Return Q1, Q3, IQR and the upper outlier fence using NumPy."""
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    return q1, q3, iqr, q3 + 1.5 * iqr

# Lambda functions = tiny one-line functions
is_canceled = lambda text: str(text).strip().lower() == "yes"    # "Yes" -> True
pct         = lambda part, whole: part / whole * 100             # percentage
cancel_rate = lambda s: s.mean() * 100                           # % of True values
round2      = lambda x: round(float(x), 2)                       # tidy rounding

print("Project 13: Hotel Booking & Occupancy Analysis")
print("Student: Ananya Mishra | Student ID: 2026B03110031")
print()

# ---------------------------------------------------------------------
# Task 1: Load and inspect
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)
show("Task 1: Dataset size", metrics(Rows=df.shape[0], Columns=df.shape[1]), index=False)
show("Task 1: First 5 records", df.head(), index=False)
show("Task 1: Column data types", pd.DataFrame({"Data_Type": df.dtypes.astype(str)}))
show("Task 1: Summary statistics", df.describe().round(2))

# ---------------------------------------------------------------------
# Task 2: Missing values and duplicate Booking_ID
# ---------------------------------------------------------------------
show("Task 2: Missing values per column", pd.DataFrame({"Missing_Count": df.isna().sum()}))
duplicates = int(df.duplicated(subset="Booking_ID").sum())
rows_before = len(df)
df = df.drop_duplicates(subset="Booking_ID", keep="first").reset_index(drop=True)
show("Task 2: Duplicate Booking_ID records",
     metrics(Duplicate_Records_Removed=duplicates, Rows_Before=rows_before, Rows_After=len(df)), index=False)

# ---------------------------------------------------------------------
# Task 3: Handle missing Lead_Time_Days and Room_Rate
# Median is used because it is not affected by extreme values.
# Room_Rate is filled with the median of its Room_Type,
# Lead_Time_Days with the median of its Booking_Channel.
# ---------------------------------------------------------------------
rate_missing = int(df["Room_Rate"].isna().sum())
lead_missing = int(df["Lead_Time_Days"].isna().sum())
df["Room_Rate"] = df["Room_Rate"].fillna(df.groupby("Room_Type")["Room_Rate"].transform("median"))
df["Lead_Time_Days"] = df["Lead_Time_Days"].fillna(df.groupby("Booking_Channel")["Lead_Time_Days"].transform("median"))
# Keep the dataset complete even if a future file contains an entirely missing group.
df["Room_Rate"] = df["Room_Rate"].fillna(df["Room_Rate"].median())
df["Lead_Time_Days"] = df["Lead_Time_Days"].fillna(df["Lead_Time_Days"].median())
show("Task 3: Missing values handled (median of group)",
     metrics(Room_Rate_Filled=rate_missing, Lead_Time_Days_Filled=lead_missing,
             Missing_Left=int(df[["Room_Rate", "Lead_Time_Days"]].isna().sum().sum())), index=False)

# ---------------------------------------------------------------------
# Task 4: Datetime, arrival month and weekday
# ---------------------------------------------------------------------
df["Checkin_Date"] = pd.to_datetime(df["Checkin_Date"])
df["Arrival_Month"] = df["Checkin_Date"].dt.month_name()
df["Arrival_Weekday"] = df["Checkin_Date"].dt.day_name()
df["Year_Month"] = df["Checkin_Date"].dt.to_period("M").astype(str)
show("Task 4: Arrival month and weekday (first 5 records)",
     df[["Booking_ID", "Checkin_Date", "Arrival_Month", "Arrival_Weekday"]].head(), index=False)

# ---------------------------------------------------------------------
# Task 5: Totals and cancellation percentage
# ---------------------------------------------------------------------
df["Is_Canceled"] = df["Canceled"].apply(is_canceled)
total_bookings = len(df)
canceled = int(df["Is_Canceled"].sum())
room_nights_booked = int(df["Nights"].sum())
room_nights_realized = int(df.loc[~df["Is_Canceled"], "Nights"].sum())
show("Task 5: Totals and cancellation percentage",
     metrics(Total_Bookings=total_bookings, Canceled_Bookings=canceled,
             Total_Room_Nights_Booked=room_nights_booked,
             Realized_Room_Nights=room_nights_realized,
             Overall_Cancellation_pct=round2(pct(canceled, total_bookings))), index=False)

# ---------------------------------------------------------------------
# Task 6: Booking revenue for non-canceled bookings
# ---------------------------------------------------------------------
df["Booking_Revenue"] = np.where(df["Is_Canceled"], 0.0, df["Nights"] * df["Room_Rate"]).round(2)
realized = df[~df["Is_Canceled"]].copy()
show("Task 6: Revenue of non-canceled bookings",
     metrics(Total_Realized_Revenue=round2(df["Booking_Revenue"].sum()),
             Average_Revenue_Per_Booking=round2(realized["Booking_Revenue"].mean())), index=False)

# ---------------------------------------------------------------------
# Task 14 (done before Task 9, which needs it): lead-time bands with pd.cut()
# ---------------------------------------------------------------------
df["Lead_Band"] = pd.cut(df["Lead_Time_Days"], bins=[-1, 7, 30, 90, np.inf],
                         labels=["Last-Minute", "Short-Term", "Medium-Term", "Advance"])
realized = df[~df["Is_Canceled"]].copy()
show("Task 14: Bookings per lead-time category (Last-Minute 0-7, Short-Term 8-30, Medium-Term 31-90, Advance 91+ days)",
     pd.DataFrame({"Bookings": df["Lead_Band"].value_counts().sort_index()}))

# ---------------------------------------------------------------------
# Task 7: Room types
# ---------------------------------------------------------------------
room = realized.groupby("Room_Type").agg(
    Realized_Bookings=("Booking_ID", "count"),
    Avg_Room_Rate=("Room_Rate", "mean"),
    Realized_Room_Nights=("Nights", "sum"),
    Total_Revenue=("Booking_Revenue", "sum")).round(2)
room["Occupancy_Share_pct"] = (room["Realized_Room_Nights"] / room["Realized_Room_Nights"].sum() * 100).round(2)
room["Revenue_Share_pct"] = (room["Total_Revenue"] / room["Total_Revenue"].sum() * 100).round(2)
room = room.reindex(["Standard", "Deluxe", "Suite"])
show("Task 7: Room-type revenue summary", room)

# ---------------------------------------------------------------------
# Task 8: Booking channel ranking
# ---------------------------------------------------------------------
channel = df.groupby("Booking_Channel").agg(Total_Bookings=("Booking_ID", "count")).join(
    realized.groupby("Booking_Channel").agg(Total_Revenue=("Booking_Revenue", "sum")),
    how="left")
channel["Total_Revenue"] = channel["Total_Revenue"].fillna(0.0)
channel["Volume_Rank"] = channel["Total_Bookings"].rank(ascending=False).astype(int)
channel["Revenue_Rank"] = channel["Total_Revenue"].rank(ascending=False).astype(int)
channel["Total_Revenue"] = channel["Total_Revenue"].round(2)
channel = channel.sort_values("Total_Revenue", ascending=False)
show("Task 8: Booking channels ranked by revenue and volume", channel)

# ---------------------------------------------------------------------
# Task 9: Cancellation by channel and lead band
# ---------------------------------------------------------------------
channel_cancel = df.groupby("Booking_Channel").agg(
    Bookings=("Booking_ID", "count"),
    Canceled=("Is_Canceled", "sum"),
    Cancellation_Rate_pct=("Is_Canceled", cancel_rate)).round(2)
channel_cancel = channel_cancel.sort_values("Cancellation_Rate_pct", ascending=False)
lead_cancel = df.groupby("Lead_Band", observed=True).agg(
    Bookings=("Booking_ID", "count"),
    Canceled=("Is_Canceled", "sum"),
    Cancellation_Rate_pct=("Is_Canceled", cancel_rate)).round(2)
show("Task 9: Cancellation rate by booking channel", channel_cancel)
show("Task 9: Cancellation rate by lead-time band", lead_cancel)

# ---------------------------------------------------------------------
# Task 10: Peak revenue months and monthly occupancy trend
# ---------------------------------------------------------------------
monthly = realized.groupby("Year_Month").agg(
    Realized_Bookings=("Booking_ID", "count"),
    Realized_Room_Nights=("Nights", "sum"),
    Revenue=("Booking_Revenue", "sum")).round(2)
monthly["Occupancy_Index"] = (monthly["Realized_Room_Nights"] / monthly["Realized_Room_Nights"].mean() * 100).round(1)
top3 = monthly["Revenue"].sort_values(ascending=False).head(3)
show("Task 10: Top 3 peak revenue months", pd.DataFrame({"Revenue": top3}))
show("Task 10: Monthly occupancy trend (Occupancy_Index 100 = average month)", monthly)

# ---------------------------------------------------------------------
# Task 11: Leisure vs business stay duration
# ---------------------------------------------------------------------
stay = realized.groupby("Traveler_Type")["Nights"].agg(["count", "mean", "median"]).round(2)
stay.columns = ["Stays", "Avg_Nights", "Median_Nights"]
show("Task 11: Average stay duration by traveler type", stay)

# ---------------------------------------------------------------------
# Task 12: NumPy statistics of daily room rates
# ---------------------------------------------------------------------
rates = df["Room_Rate"].to_numpy()
show("Task 12: NumPy statistics of daily room rate",
     metrics(Mean=round2(np.mean(rates)), Median=round2(np.median(rates)),
             Std_Deviation=round2(np.std(rates))), index=False)

# ---------------------------------------------------------------------
# Task 13: Unusually high values (IQR method)
# ---------------------------------------------------------------------
q1, q3, iqr, fence = iqr_upper_fence(realized["Booking_Revenue"].to_numpy())
high_revenue = realized[realized["Booking_Revenue"] > fence]
q1r, q3r, iqr_r, fence_r = iqr_upper_fence(rates)
high_rate = df[df["Room_Rate"] > fence_r]
iqr_table = pd.DataFrame({
    "Q1": [round2(q1), round2(q1r)], "Q3": [round2(q3), round2(q3r)],
    "IQR": [round2(iqr), round2(iqr_r)], "Upper_Fence": [round2(fence), round2(fence_r)],
    "Unusually_High_Count": [len(high_revenue), len(high_rate)]},
    index=["Booking_Revenue", "Room_Rate"])
show("Task 13: IQR outlier limits", iqr_table)
show("Task 13: Top 10 unusually high revenue bookings",
     high_revenue.sort_values("Booking_Revenue", ascending=False)[
         ["Booking_ID", "Room_Type", "Nights", "Room_Rate", "Booking_Revenue"]].head(10), index=False)

# ---------------------------------------------------------------------
# Task 15: Five recommendations (numbers come from the results above)
# ---------------------------------------------------------------------
top_c, low_c = channel_cancel.index[0], channel_cancel.index[-1]
top_band = lead_cancel["Cancellation_Rate_pct"].idxmax()
rpn = room["Total_Revenue"] / room["Realized_Room_Nights"]          # revenue per room night
best_room, worst_room = rpn.idxmax(), rpn.idxmin()
low_month = monthly["Realized_Room_Nights"].idxmin()
share = channel["Total_Revenue"] / channel["Total_Revenue"].sum() * 100

recs = [
    f"1. Reduce {top_c} cancellations ({channel_cancel.loc[top_c, 'Cancellation_Rate_pct']:.1f}% vs {channel_cancel.loc[low_c, 'Cancellation_Rate_pct']:.1f}% for {low_c}): use deposits or stricter cancellation windows.",
    f"2. Manage '{top_band}' lead-time bookings ({lead_cancel.loc[top_band, 'Cancellation_Rate_pct']:.1f}% cancel): send reminders and use controlled overbooking.",
    f"3. Upsell to {best_room} rooms: {rpn[best_room]:,.0f} per room night, {rpn[best_room] / rpn[worst_room]:.1f} times a {worst_room} room.",
    f"4. Add staff and raise rates in peak months ({', '.join(top3.index)}); run promotions in the weakest month ({low_month}).",
    f"5. Grow direct bookings: Direct Website gives {share['Direct Website']:.1f}% of revenue (OTA {share['OTA']:.1f}%) with a lower cancellation rate.",
]
print("Task 15: Five revenue and operations recommendations")
for r in recs:
    print(r)
print()
results.append(("Task 15: Five revenue and operations recommendations",
                pd.DataFrame({"Recommendation": recs}), False))

# ---------------------------------------------------------------------
# Save: cleaned dataset + ONE results sheet (all tables stacked in one CSV file)
# ---------------------------------------------------------------------
df.to_csv(CLEAN_FILE, index=False)

SHEET_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(SHEET_FILE, "w", newline="") as sheet:
    sheet.write("Project 13: Hotel Booking & Occupancy Analysis | Student: Ananya Mishra | Student ID: 2026B03110031\n\n")
    for title, table, keep_index in results:
        sheet.write(title + "\n")
        table.to_csv(sheet, index=keep_index, lineterminator="\n")
        sheet.write("\n")

print("Done. Results sheet saved to", SHEET_FILE)
