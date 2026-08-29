# Data needed from the paper

You only need one source of ground-truth data: daily CO₂ emissions and gross load per plant, for all 1303 plants in the contiguous US, across the four study periods.

The source is EPA CAMPD.

## Tuesday 19/05/2026

[ ] Download EPA CAMPD relevant data for the four study periods in `scripts/download_epa.py`

- [x] Write a small notebook that fetches one file directly from the bulk-files API to understand the structure
- [x] Write a small notebook which downloads parquet data with `pudl` project. `pudl` is justified cause it provides clean data
- [x] Load EPA CEMS data for 2021 and 2022 into a notebook and aggregate the data as the authors of the paper did

## Plant Count Discrepancy vs. Original Paper

After applying all filters explicitly stated in Mo et al. (2025) Section 2.1.1 and 2.3.1:

- Source: EPA Air Markets Program Data daily emissions files (campd.epa.gov)
- Geographic: Contiguous US only (excludes AK, HI, PR, VI, GU, MP, AS)
- Fuel type: Fossil primary fuel only (excludes Wood-primary biomass plants)
- Operating: Positive CO₂ and gross load on the day
- Aggregation: Unit-level data summed to facility-day using EPA Facility ID

Our resulting valid plant counts per period:

| Period                | Our count | Paper's count | Difference |
| --------------------- | --------- | ------------- | ---------- |
| 2021-04-01 to 2021-05 | 1039      | 583           | +456       |
| 2021-09-01 to 2021-10 | 1067      | 590           | +477       |
| 2022-04-01 to 2022-04 | 987       | 513           | +474       |
| 2022-09-01 to 2022-09 | 1047      | 592           | +455       |

We were unable to recover the paper's exact plant count from its stated methodology.
The residual gap likely reflects undocumented filtering criteria. We retain the broader universe for training as it provides more diverse
samples and the model methodology is independent of dataset size. Evaluation metrics will surely differ in magnitude from the paper's reported values for this reason.

## Sunday 31/05/2026

[ ] Closing out phase 1 of the project

- [x] Build the facility attributes used in the paper and extracted from EPA
  - Step 1: Find the Facility files in the listing - In a separate notebook called `facilities` explore what a `Facility` file looks like
  - Step 2: Download the relevant Facility files through adding a function called `select_facility_files` in `src/co2sat/data/epa.py` script
  - Step 3: Explore downloaded facilities through a separate notebook `explore_downloaded_facilities`
  - Step 4: Parse generation capacities of power plants and create functions to filter the operating ones
  - Step 5: Define Module-level constants and helpers based on the discovery of the facility files
  - Step 6: Implement the functions for loading, filtering and aggregating the static attributes found in facility files with epa files downloaded previously. Functions are written in `src/co2sat/data/epa.py` script
  - Step 7: Run the consolidation prepared in previous step in a newly created script `scripts/consolidate_facilities.py`
  - Step 8: Conduct sanity checks in the notebook `explore_downloaded_facilities`
  - Step 9: Join the aggregated facility parquet file to daily emissions through the script `join_daily_facilities`
  - Step 10: Verify Per-Period Plant Counts since the previous script joins over 2021 and 2022 entirely. Operation done in the notebook `explore_downloaded_facilities`

Our resulting valid plant counts per period after joining the EPA attributes of table 3 and applying the appropriate filters are:

| Period                | Plants | (paper) | Δ    | Rows  |
| --------------------- | ------ | ------- | ---- | ----- |
| 2021-04-01 to 2021-05 | 1038   | 583     | +455 | 29525 |
| 2021-09-01 to 2021-10 | 1066   | 590     | +476 | 21696 |
| 2022-04-01 to 2022-04 | 986    | 513     | +473 | 17607 |
| 2022-09-01 to 2022-09 | 1045   | 592     | +453 | 21405 |

## Monday 01/06/2026

[x] Closing out phase 1 of the project

- [x] Operate EDA on the joined dataset and plot emissions distribution, intensity by fuel, geographic map, top emitters and reporting completeness
  - [x] Plot emissions distribution
  - [x] Plot emissions intensity by fuel
  - [x] Plot geographic distribution
  - [x] Detect top emitters and report completeness

## Tuesday 02/06/2026

[ ] Starting phase 2 of the project: GOES-16 Satellite Pipeline. The ultimate goal of this phase is to produce a dynamical feature matrix of size 16 × 24 of spectral bands for each facility of the previous phase.

- [ ] Extract one pixel from one file and start understanding the data
  - [x] AWS CLI Anonymous Access to S3 bucket of satellite data:
        The good news are GOES16 data can be accessed from aws s3 without having an account in aws; just from the cli. This is why `awscli` should be installed.
        The paper collected data from Advanced Baseline Imager : This is the primary instrument on the GOES-R Series for imaging Earth’s weather, ocean and environment.
        A quick command to visualize the different products of goes16 gives the following:

```bash
  $ aws s3 ls --no-sign-request s3://noaa-goes16/ | head
        PRE ABI-L1b-RadC-Reproc/
        PRE ABI-L1b-RadC/
        PRE ABI-L1b-RadF-Reproc/
        PRE ABI-L1b-RadF/
        PRE ABI-L1b-RadM/
        PRE ABI-L2-ACHA2KMC/
        PRE ABI-L2-ACHA2KMF/
        PRE ABI-L2-ACHA2KMM/
        PRE ABI-L2-ACHAC/
        PRE ABI-L2-ACHAF/
```

For direct retrieval of CO2 using raw, calibrated top-of-atmosphere radiances as stated by the paper, Level 1B (L1B) data is required. Researching in the [goes16 github docs](https://github.com/awslabs/open-data-docs/blob/main/docs/noaa/noaa-goes16/README.md) shows that `ABI-L1b-RadC` is the most appropriate product for this task.

- [x] Understand file hierarchy
      Giving a specific time and date through the following command can show the file naming structure

```bash
$ aws s3 ls --no-sign-request s3://noaa-goes16/ABI-L1b-RadC/2021/091/12/ | head -50
2021-04-01 13:04:09    7171404 OR_ABI-L1b-RadC-M6C01_G16_s20210911201147_e20210911203520_c20210911203556.nc
2021-04-01 13:09:09    7428500 OR_ABI-L1b-RadC-M6C01_G16_s20210911206147_e20210911208520_c20210911208562.nc
2021-04-01 13:14:13    7690778 OR_ABI-L1b-RadC-M6C01_G16_s20210911211147_e20210911213520_c20210911213554.nc
2021-04-01 13:19:13    7943477 OR_ABI-L1b-RadC-M6C01_G16_s20210911216147_e20210911218520_c20210911218555.nc
2021-04-01 13:24:15    8191588 OR_ABI-L1b-RadC-M6C01_G16_s20210911221147_e20210911223520_c20210911223556.nc
2021-04-01 13:29:16    8439037 OR_ABI-L1b-RadC-M6C01_G16_s20210911226147_e20210911228520_c20210911228559.nc
2021-04-01 13:34:07    8684867 OR_ABI-L1b-RadC-M6C01_G16_s20210911231147_e20210911233520_c20210911233557.nc
2021-04-01 13:39:05    8935570 OR_ABI-L1b-RadC-M6C01_G16_s20210911236147_e20210911238520_c20210911238564.nc
2021-04-01 13:44:19    9188838 OR_ABI-L1b-RadC-M6C01_G16_s20210911241147_e20210911243520_c20210911243555.nc
2021-04-01 13:49:06    9422562 OR_ABI-L1b-RadC-M6C01_G16_s20210911246147_e20210911248520_c20210911248556.nc
2021-04-01 13:54:17    9670957 OR_ABI-L1b-RadC-M6C01_G16_s20210911251147_e20210911253520_c20210911253563.nc
2021-04-01 13:59:25    9906595 OR_ABI-L1b-RadC-M6C01_G16_s20210911256147_e20210911258520_c20210911258561.nc
2021-04-01 13:04:27   55966422 OR_ABI-L1b-RadC-M6C02_G16_s20210911201147_e20210911203520_c20210911203548.nc
2021-04-01 13:09:17   56604120 OR_ABI-L1b-RadC-M6C02_G16_s20210911206147_e20210911208520_c20210911208545.nc
2021-04-01 13:14:28   57251485 OR_ABI-L1b-RadC-M6C02_G16_s20210911211147_e20210911213520_c20210911213545.nc
2021-04-01 13:19:29   57883635 OR_ABI-L1b-RadC-M6C02_G16_s20210911216147_e20210911218520_c20210911218546.nc
2021-04-01 13:24:29   58537841 OR_ABI-L1b-RadC-M6C02_G16_s20210911221147_e20210911223520_c20210911223546.nc
2021-04-01 13:29:31   59176512 OR_ABI-L1b-RadC-M6C02_G16_s20210911226147_e20210911228520_c20210911228546.nc
2021-04-01 13:34:17   59838917 OR_ABI-L1b-RadC-M6C02_G16_s20210911231147_e20210911233520_c20210911233546.nc
2021-04-01 13:39:32   60497385 OR_ABI-L1b-RadC-M6C02_G16_s20210911236147_e20210911238520_c20210911238545.nc
2021-04-01 13:44:34   61173309 OR_ABI-L1b-RadC-M6C02_G16_s20210911241147_e20210911243520_c20210911243545.nc
2021-04-01 13:49:34   61817921 OR_ABI-L1b-RadC-M6C02_G16_s20210911246147_e20210911248520_c20210911248544.nc
2021-04-01 13:54:30   62469522 OR_ABI-L1b-RadC-M6C02_G16_s20210911251147_e20210911253520_c20210911253548.nc
2021-04-01 13:59:39   63103063 OR_ABI-L1b-RadC-M6C02_G16_s20210911256147_e20210911258520_c20210911258546.nc
2021-04-01 13:04:13    9184874 OR_ABI-L1b-RadC-M6C03_G16_s20210911201147_e20210911203520_c20210911203558.nc
2021-04-01 13:09:08    9441068 OR_ABI-L1b-RadC-M6C03_G16_s20210911206147_e20210911208520_c20210911208556.nc
2021-04-01 13:14:18    9695896 OR_ABI-L1b-RadC-M6C03_G16_s20210911211147_e20210911213520_c20210911213560.nc
2021-04-01 13:19:22    9946572 OR_ABI-L1b-RadC-M6C03_G16_s20210911216147_e20210911218520_c20210911218561.nc
2021-04-01 13:24:20   10192821 OR_ABI-L1b-RadC-M6C03_G16_s20210911221147_e20210911223520_c20210911223560.nc
2021-04-01 13:29:22   10446576 OR_ABI-L1b-RadC-M6C03_G16_s20210911226147_e20210911228520_c20210911228564.nc
2021-04-01 13:34:07   10706447 OR_ABI-L1b-RadC-M6C03_G16_s20210911231147_e20210911233520_c20210911233569.nc
2021-04-01 13:39:05   10968100 OR_ABI-L1b-RadC-M6C03_G16_s20210911236147_e20210911238520_c20210911238561.nc
2021-04-01 13:44:25   11217853 OR_ABI-L1b-RadC-M6C03_G16_s20210911241147_e20210911243520_c20210911243559.nc
2021-04-01 13:49:24   11464771 OR_ABI-L1b-RadC-M6C03_G16_s20210911246147_e20210911248520_c20210911248561.nc
2021-04-01 13:54:17   11714667 OR_ABI-L1b-RadC-M6C03_G16_s20210911251147_e20210911253520_c20210911253565.nc
2021-04-01 13:59:03   11950222 OR_ABI-L1b-RadC-M6C03_G16_s20210911256147_e20210911258520_c20210911258554.nc
2021-04-01 13:04:09    1864033 OR_ABI-L1b-RadC-M6C04_G16_s20210911201147_e20210911203520_c20210911203551.nc
2021-04-01 13:09:05    1878304 OR_ABI-L1b-RadC-M6C04_G16_s20210911206147_e20210911208520_c20210911208551.nc
2021-04-01 13:14:08    1892067 OR_ABI-L1b-RadC-M6C04_G16_s20210911211147_e20210911213520_c20210911213551.nc
2021-04-01 13:19:11    1905499 OR_ABI-L1b-RadC-M6C04_G16_s20210911216147_e20210911218520_c20210911218552.nc
2021-04-01 13:24:11    1920995 OR_ABI-L1b-RadC-M6C04_G16_s20210911221147_e20210911223520_c20210911223552.nc
2021-04-01 13:29:13    1934644 OR_ABI-L1b-RadC-M6C04_G16_s20210911226147_e20210911228520_c20210911228550.nc
2021-04-01 13:34:04    1950341 OR_ABI-L1b-RadC-M6C04_G16_s20210911231147_e20210911233520_c20210911233550.nc
2021-04-01 13:39:02    1967291 OR_ABI-L1b-RadC-M6C04_G16_s20210911236147_e20210911238520_c20210911238551.nc
2021-04-01 13:44:14    1984344 OR_ABI-L1b-RadC-M6C04_G16_s20210911241147_e20210911243520_c20210911243550.nc
2021-04-01 13:49:03    2003711 OR_ABI-L1b-RadC-M6C04_G16_s20210911246147_e20210911248520_c20210911248550.nc
2021-04-01 13:54:10    2021629 OR_ABI-L1b-RadC-M6C04_G16_s20210911251147_e20210911253520_c20210911253554.nc
2021-04-01 13:59:01    2040793 OR_ABI-L1b-RadC-M6C04_G16_s20210911256147_e20210911258520_c20210911258549.nc
2021-04-01 13:04:20    9177855 OR_ABI-L1b-RadC-M6C05_G16_s20210911201147_e20210911203521_c20210911203583.nc
2021-04-01 13:09:07    9381862 OR_ABI-L1b-RadC-M6C05_G16_s20210911206147_e20210911208520_c20210911208558.nc
```

This shows several abbreviations explained in the [naming convention](https://cimss.ssec.wisc.edu/goes/ABI_File_Naming_Conventions.pdf). Where:

- `OR_ABI-L1b-RadC` stands for operational sensor, ABI instrument, level 1b, reflectance radiance of CONUS
- `M6C01` stands for Mode 6 (i.e: the way and the frequency by which the sensor scans the CONUS region and the western hemisphere) and Channel 1 (i.e: the number of the band which refers the wavelength of signal that the satellite's instrument is tuned to detect. In our case, there are 16 channels, hence the name GOES16)
- `G16` stands for GOES16
- `s20210911201147` stands for the start time of scan in the unix system notation
- `e20210911203520` stands for the end time of scan in the unix system notation
- `c20210911203556` stands for the creation time of scan in the unix system notation

Counting the total number of scans per hour must then yield: `Number of bands * Number of scans per band per hour = 16 * 12 = 192 scans/hour`. Therefore, there must be 192 files generated per hour by `ABI-L1b-RadC` instrument of GOES16 satellite. This number is confirmed by the following command

```bash
$ aws s3 ls --no-sign-request s3://noaa-goes16/ABI-L1b-RadC/2021/091/12/ | wc -l
192
```

- [x] Download one file by hand
      We chose to download the file `OR_ABI-L1b-RadC-M6C01_G16_s20210911201147_e20210911203520_c20210911203556.nc` by running the following commands

```bash
$ mkdir -p data/raw/goes16_test
$ aws s3 cp --no-sign-request s3://noaa-goes16/ABI-L1b-RadC/2021/091/12/ \ OR_ABI-L1b-RadC-M6C01_G16_s20210911201147_e20210911203520_c20210911203556.nc \
data/raw/ goes16_test/
```

- [x] Inspect the downloaded file
      Inspection of the downloaded file is proceeded in the notebook `06_satellite_file_exploration.ipynb`
- [x] Try coordinate transformation
      GOES16 is geostationary satellite, which means to extract a pixel at a specific (lat, lon), you need to convert that lat/lon to (x,y) and each pixel corresponds to a fixed (x,y) coordinate.
      The transformation formulae are specified in section 5.1.2.8 of [volume 3 GOES R SERIES PRODUCT DEFINITION AND USERS’ GUIDE (PUG)](https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf). Python library called `pyproj` is used in order to use these formulae
      Example of transformation is displayed in notebook `07_coordinates_transformation.ipynb`
- [ ] Convert all images to html files in order to upload notebooks to github with ease

## Tuesday 25/08/2026

- [x] Sub-Phase 2.2: single plant-day extraction validated (W A Parish 16×24
      matrix, physics checks: diurnal arc, water-vapor band ordering, thermal lag)
- [x] Sub-Phase 2.3: batched resumable extractor (one file open serves all
      1,158 plants; verified numerically identical to single-plant path)
- [x] Sub-Phase 2.4: full extraction EXECUTED on Google Colab.
      Local pilot: 24 scan-hours in 34.5 min (bandwidth-bound ~6.5 Mbps)
      → projected 85 h locally → moved compute to the data.
      Colab run: 3,251 done + 12 missing (GOES outages) + 0 failed in 81 min,
      4 processes (ProcessPoolExecutor after HDF5 thread-safety segfault
      with threads). Cache: 3,252 parquets, merged locally.

## Thursday 27/08/2026

- [x] Sub-Phase 2.5: consolidation + quality checks (notebook 09)
      DELIVERABLE: data/processed/dynamic_features.parquet — 3,765,312 rows
      (1,158 plants × 136 days × 24h − 12 missing scan-hours), sorted by
      facility/date/hour, snappy compression.

      COMPLETENESS (file level): 98.5% of plant-days have all 24 hours. The 12
      missing scan-hours = exactly 2 GOES-16 outages — 2021-04-29 H21-H22 and
      2022-09-13 H11-H20 — matching the extraction log.

      MASKED PIXELS (value level): ~0.4-0.5% NULLs per band. (Note: pandas NaN
      becomes Parquet NULL via pyarro) Two causes: (i) six whole-scan masked events (2021-05-18 H14,
      2022-04-25 H22, 2022-04-26 H19-H20, 2022-09-27 H12+H18); (ii) recurring
      H05 UTC masking on early-April/early-September dates — satellite local
      midnight during geostationary eclipse seasons (solar-intrusion QC
      masking). No chronic bad-pixel plants (max 34 rows/plant, all 1,158
      affected uniformly) → no plant exclusions.

      LABEL COVERAGE (period-filtered): 90,261 labeled plant-days in-period,
      100% with satellite data, 88,795 with full 24h. Per-period vs paper:
      P1 28,838/20,306 (1.42) · P2 21,696/14,464 (1.50) · P3 17,607/11,213
      (1.57) · P4 20,654/15,243 (1.35) — uniform ratios, consistent with the
      documented plant-universe surplus from Phase 1.

      GAP RULE (applied at Phase 3 matrix assembly; this parquet stays raw):
      per (facility, date, band), absent values = missing hours + NULLs.
      ≤2 absent → linear interpolation along the hour axis (nearest-value fill
      at day boundaries); >2 → drop the plant-day.
      Effect: 2021-04-29 kept (2h bridge); 2022-09-13 dropped (779 labeled
      samples); 261 plant-days with 3 NULL hours dropped (~150 labeled); all
      other events interpolated. Estimated usable samples ≈ 89.3K (paper:
      61,226).

      DEVIATION LOG: the paper's periods sit in the same eclipse seasons and
      faced the same masked hours and outages; it documents no gap handling.
      This rule is an explicit interpretation.

- [x] Sub-Phase 2.6: dynamic features EDA (notebook 10)
      Distributions: reflectance bimodal (night mass at 0), thermals 250-310 K.
      Fleet diurnal cycle consistent with single-plant physics across 4 time zones.
      April/September shift measurable in window + water-vapor bands. Window thermals near-collinear (r>0.95):
      ~16 bands ≪ 16 independent signals. Naive band_07 vs CO2 correlation:
      r = -0.05 — the univariate baseline the model must beat.

## PHASE 2 COMPLETE

Deliverable: data/processed/dynamic_features.parquet (3.77M rows, 1.158 plants × 136 days × 24 h × 16 bands, QC'd, gap rule documented).

## Saturday 29/08/2026 — Phase 3: Static Features

- [x] Session 1: satellite zenith angle + EPA statics assembly
      Zenith angle implemented per paper eqs (1)-(3) with paper constants
      (SatLon=-75.2°, R=6370 km, r=42156 km — paper's values kept verbatim).
      Verified against an independent implementation: W A Parish = 41.033°,
      Miami 30.668°, Denver 55.245°, Seattle 70.885°; 5 tests passing incl.
      subsatellite zero, east-west symmetry, vectorization consistency.
      static_features.parquet created including: 1,158 facilities × 9 cols (capacity,lat/lon, 4 fuel ratios, zenith). QC gates: 0 NaN, fuel ratios sum to 1, all facilities within CONUS bounds. Zenith gradient map exported
      (smooth SE→NW sweep confirms geometry).

- [x] Session 2: altitude
      Source substitution (deviation log): paper cites Mapzen terrain tiles
      [33]; Mapzen is defunct → USGS EPQS point queries (both SRTM-derived;
      differences at plant coordinates negligible vs 2-km satellite pixels).
      1,158 point queries via resumable cache (data/interim/altitude_cache.
      parquet). QC: 0 missing after retries, range [min] to [max] m, coastal
      plants ≈ 0 m as expected. Merged into static_features.parquet.

- [x] Phase 3 Session 3: EDGAR surroundings
      Source: EDGAR v8.0 [35], CO2 excl. short-cycle, sector TOTALS, year 2021,
      annual gridmap 0.1° (ton/cell/yr). Extraction: containing-cell value
      (paper leaves 'surrounding' unquantified — interpretation documented;
      3×3-window robustness: r = [0,78]). Self-inclusion note: EDGAR cells contain
      the plants' own inventoried point-source emissions — feature partly
      encodes typical plant magnitude; replicated as-is (paper could not
      exclude it either).

- [x] Session 4: consumption surroundings
      Source substitution (deviation log #5): paper cites Hu et al. 2022 [34]
      (nightlight-derived global EPC 1992-2019); no public deposit exists → Chen et al. 2022 (Sci Data, figshare 17004523) substituted: same phenomenon (1 km gridded electricity consumption from calibrated nightlights, 1992-2019). Year 2019 used (latest available = closest to study period). Raster in ESRI:54009 (Mollweide); plant coordinates transformed via pyproj (always_xy)
      before sampling — probed against known city centers (Houston,
      Manhattan, Chicago >> rural Nevada/Texas/Montana ≈ 0) and top/bottom
      plant rankings (urban vs rural as expected). Containing 1-km cell;
      dark/nodata cells filled with 0 (nightlight-dark = negligible local
      consumption). Cross-check vs EDGAR feature: r(log-log) = 0.22 — weak value, explained by the ~100:1 cell-area mismatch (0.1° vs
      1 km) and EDGAR's point-source self-inclusion; the two features are
      treated as complementary neighborhood proxies.
      static_features.parquet complete: 1,158 facilities × 11 features,
      matching paper Table 3.
