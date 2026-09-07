from pathlib import Path

from app.preprocessing.pmgsy import PMGSYIngestor


# =========================================================
# CONFIGURATION
# =========================================================

PMGSY_ROOT = Path(
    r"D:\rag_types\NER\pmGSY"
)

OUTPUT_FILE = Path(
    "data/processed/roads.parquet"
)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("NER PMGSY PREPROCESSING")
    print("=" * 60)

    ingestor = PMGSYIngestor(
        PMGSY_ROOT
    )

    roads = ingestor.process()

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\nPhase distribution:")
    print(
        roads["pmgsy_phase"]
        .value_counts()
    )

    print("\nState distribution:")
    print(
        roads["state_name"]
        .value_counts()
    )

    print("\nSurface distribution:")
    print(
        roads["surface_type"]
        .value_counts()
        .head(20)
    )

    print("\nTotal road length:")
    print(
        f"{roads['length_km'].sum():,.2f} km"
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    ingestor.save(
        roads,
        OUTPUT_FILE
    )

    print("\nPREPROCESSING COMPLETE")


if __name__ == "__main__":
    main()