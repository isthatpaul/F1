import fastf1
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")


def load_season_results(year, cache_dir="cache"):
    """
    Load all race results for a given F1 season.

    Args:
        year:  Season year (e.g., 2023, 2024)
        cache_dir: Path to FastF1 cache directory

    Returns:
        DataFrame with all race results
    """
    # Enable cache
    cache_path = os.path.abspath(cache_dir)
    fastf1.Cache.enable_cache(cache_path)

    print(f"Loading {year} season.. .\n")

    # Get season schedule
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        raise ValueError(f"Could not load {year} schedule: {e}")

    all_results = []

    # Loop through each race - use enumerate instead of iterrows to avoid type issues
    for race_idx, (idx, race) in enumerate(schedule.iterrows(), start=1):
        # Skip testing events
        event_format = race.get("EventFormat", "")
        if "testing" in str(event_format).lower():
            continue

        # Use race_idx which is guaranteed to be an int
        race_name = race.get("EventName", f"Round {race_idx}")
        round_number = race.get("RoundNumber", race_idx)

        print(f"  Round {round_number}: {race_name}...", end=" ")

        try:
            # Load race session
            session = fastf1.get_session(year, round_number, "R")
            session.load()

            # Check if results exist
            if session.results is None or len(session.results) == 0:
                print("❌ No results")
                continue

            # Get results
            results = session.results.copy()

            # Add race metadata
            results["Race"] = race_name
            results["Round"] = round_number
            results["Year"] = year
            results["Date"] = race.get("EventDate", pd.NaT)

            # Standardize column names - handle both 2023 and 2024 formats
            if "FullName" in results.columns:
                results["DriverFull"] = results["FullName"]
            elif "BroadcastName" in results.columns:
                results["DriverFull"] = results["BroadcastName"]
            else:
                print("⚠️ No driver names")
                continue

            if "Abbreviation" in results.columns:
                results["Driver"] = results["Abbreviation"]
            else:
                driver_num = results.get("DriverNumber")
                results["Driver"] = str(driver_num) if driver_num is not None else "UNK"

            if "TeamName" in results.columns:
                results["Team"] = results["TeamName"]
            else:
                results["Team"] = "Unknown"

            # Convert numeric columns safely
            results["Position"] = pd.to_numeric(
                results.get("Position", pd.Series([None] * len(results))),
                errors="coerce",
            )
            results["GridPosition"] = pd.to_numeric(
                results.get("GridPosition", pd.Series([None] * len(results))),
                errors="coerce",
            )
            results["Points"] = pd.to_numeric(
                results.get("Points", pd.Series([0] * len(results))), errors="coerce"
            )

            # Add Status if missing
            if "Status" not in results.columns:
                results["Status"] = "Finished"

            # Add Time if missing
            if "Time" not in results.columns:
                results["Time"] = pd.NaT

            all_results.append(results)
            print("✓")

        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            continue

    if not all_results:
        raise ValueError(f"No race data loaded for {year}")

    # Combine all races
    season_df = pd.concat(all_results, ignore_index=True)

    # Select relevant columns
    columns_to_keep = [
        "Year",
        "Round",
        "Race",
        "Date",
        "DriverFull",
        "Driver",
        "Team",
        "Position",
        "GridPosition",
        "Points",
        "Status",
        "Time",
    ]

    # Keep only columns that exist
    columns_to_keep = [col for col in columns_to_keep if col in season_df.columns]
    season_df = season_df[columns_to_keep]

    # Remove duplicates
    season_df = season_df.drop_duplicates()

    print(f"\n✓ Loaded {len(season_df)} race results from {year} season")
    print(f"  Drivers: {season_df['DriverFull'].nunique()}")
    print(f"  Races: {season_df['Race'].nunique()}")

    return season_df


def load_multiple_seasons(years, cache_dir="cache"):
    """
    Load multiple seasons of F1 data.

    Args:
        years: List of years (e.g., [2023, 2024])
        cache_dir: Path to FastF1 cache

    Returns:
        Combined DataFrame
    """
    all_seasons = []

    for year in years:
        print(f"\n{'='*60}")
        print(f"Loading {year} Season")
        print(f"{'='*60}")

        try:
            season_df = load_season_results(year, cache_dir)
            all_seasons.append(season_df)
            print(f"✓ {year} loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load {year}: {e}")
            continue

    if not all_seasons:
        raise ValueError("No seasons loaded successfully")

    combined_df = pd.concat(all_seasons, ignore_index=True)

    print(f"\n{'='*60}")
    print("COMBINED DATA SUMMARY")
    print(f"{'='*60}")
    print(f"  Total results: {len(combined_df)}")
    print(f"  Years: {sorted(combined_df['Year'].unique())}")
    print(f"  Total drivers:  {combined_df['DriverFull'].nunique()}")
    print(f"  Total races:  {combined_df['Race'].nunique()}")
    print(f"{'='*60}\n")

    return combined_df


def load_all_available_data(cache_dir="cache"):
    """
    Load all available F1 data (2023 and 2024).

    Convenience function that automatically loads both seasons.

    Args:
        cache_dir: Path to FastF1 cache

    Returns:
        Combined DataFrame with both seasons
    """
    print("Loading all available F1 data (2023-2024)...\n")
    return load_multiple_seasons([2023, 2024], cache_dir)


def quick_test(year=2024):
    """
    Quick test function to verify data loading.

    Args:
        year: Year to test (default:  2024)
    """
    print(f"Testing {year} data loading...\n")

    try:
        df = load_season_results(year)

        print(f"\n✓ Test successful!")
        print(f"\nSample data:")
        cols_to_show = ["Round", "Race", "DriverFull", "Team", "Position", "Points"]
        available_cols = [col for col in cols_to_show if col in df.columns]
        print(df.head(10)[available_cols])

        print(f"\n{year} Championship Standings:")
        standings = (
            df.groupby("DriverFull")["Points"].sum().sort_values(ascending=False)
        )
        print(standings.head(10))

        return df

    except Exception as e:
        print(f"✗ Test failed:  {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # When run directly, test both seasons
    print("=" * 60)
    print("TESTING LOAD_FASTF1.PY")
    print("=" * 60)

    # Test 2023
    print("\n[TEST 1] Loading 2023...")
    test_2023 = quick_test(2023)

    # Test 2024
    print("\n\n[TEST 2] Loading 2024...")
    test_2024 = quick_test(2024)

    # Test loading both
    if test_2023 is not None and test_2024 is not None:
        print("\n\n[TEST 3] Loading both seasons together...")
        combined = load_all_available_data()
        print("\n✓ All tests passed!")
    else:
        print("\n⚠ Some tests failed")
