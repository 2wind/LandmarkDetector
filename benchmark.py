import os
import pandas as pd
import subprocess
import time

input_folder = "AutoAlign_test/"
output_folder = "Benchmark/"


valid_prefix = ['40374___________000_lat', 'B4867___________000_lat', 'B7524___________000_lat', 'B13616___________000_lat', '25160___________000_lat', 'B25611___________000_lat', 'B4151___________000_lat', '19951816___________000_lat', '48301___________000_lat', 'B16939___________000_lat', 'B17545___________000_lat', 'B5080___________000_lat', '40275___________000_lat', '0101___________000_lat', 'B10243___________000_lat', 'B15809___________000_lat', '17285___________000_lat', '12488___________000_lat', 'B25776___________000_lat', '14644377___________000_lat', 'B22163___________000_lat', 'B15167___________000_lat', '22248198___________000_lat', 'B12007___________000_lat', 'B15955___________000_lat', '01042185___________3618_lat', 'B8753___________000_lat', '48329___________000_lat', 'B17915___________000_lat', 'B23418___________000_lat', 'B19358___________000_lat', 'B19336___________000_lat', 'B13247___________000_lat', 'B7887___________000_lat', 'B22050___________000_lat', '23987___________000_lat', 'B19857___________000_lat', '45325___________000_lat', 'B10606___________000_lat', 'B19011___________000_lat', 'B4667___________000_lat', 'B9871___________000_lat', 'B6863___________000_lat', 'B11375___________000_lat', 'B12007___________001_lat', 'B14138___________000_lat', 'B23013___________000_lat', '28800___________000_lat', '43476___________000_lat', '21706747___________000_lat']
def return_path (common_path):
    pi, pt, fi, ft = "_photo.jpg", "_photo.txt", "_film.jpg", "_film.txt"
    image_path = common_path + pi
    tsv_path = common_path + pt
    film_path = common_path + ft
    film_img_path = common_path + fi

    return image_path, tsv_path, film_img_path, film_path

times = pd.DataFrame(columns=['prefix', 'time'])

for prefix in valid_prefix:
    pi, pt, fi, ft = return_path(prefix)
    start = time.perf_counter()
    subprocess.run(f"parse -v -m resnet_best_model.pth -fi {input_folder + fi} -ft {input_folder + ft} -pi {input_folder + pi} -pt {input_folder + pt} -o {output_folder+prefix}.txt --output_image {output_folder+prefix}.jpg --debug_text {output_folder+prefix}.csv" )
    end = time.perf_counter()
    print(end - start)
    times.loc[len(times.index)] = [prefix, end - start]

print(times)
times.to_csv("bench.csv")