# this program loads a thermal image of a guassian spot profile and a given scale in mm per pixel then calculates the FWHM of the spot profile at the 1/e^2 points

import os
import argparse
from PIL import Image
import scipy.optimize as opt
import numpy as np
from collections import namedtuple

IMAGE_PATH = "image_crop.jpg"
SCALE_PATH = "scale.txt"

# define a named tuple to hold gaussian parameters
GaussianParameters = namedtuple("GaussianParameters", ["A", "mux", "muy", "sigmax", "sigmay", "offset"])

def gaussian(location, A, mux, muy, sigmax, sigmay, offset):
    x, y = location
    z_x = (1 / (sigmax * np.sqrt(2 * np.pi))) * np.exp(-1 * ((x - mux)**2 / (2 * sigmax**2)))
    z_y = (1 / (sigmay * np.sqrt(2 * np.pi))) * np.exp(-1 * ((y - muy)**2 / (2 * sigmay**2)))
    g = (A * z_x * z_y) + offset
    return g.ravel()

def main():
    # load image and scale data
    print(f"Loading data from {args.case}... ", end="")
    image = Image.open(f"{args.case}/{IMAGE_PATH}", "r")
    image_format = image.format
    with open(f"{args.case}/{SCALE_PATH}", "r") as scale:
        scale = float(scale.read())
    print("complete")

    # convert image to grayscale
    image = image.convert("L")

    # convert image to numpy array
    image_data = np.array(image)

    # create a reasonable initial guess for the gaussian parameters
    initial_guess = GaussianParameters(
        A = np.max(image) - np.min(image),
        mux = 0.5,
        muy = 0.5,
        sigmax = 1,
        sigmay = 1,
        offset = np.min(image)
    )

    # create a meshgrid of x and y values that matches the image size
    numx = image_data.shape[1]
    numy = image_data.shape[0]
    aspect_ratio = numy / numx
    x = np.linspace(0, 1.0, numx)
    y = np.linspace(0, 1.0 * aspect_ratio, numy)
    x, y = np.meshgrid(x, y)

    # show the initial guess as an image with data normalized to 0-255
    initial_guess_data = gaussian((x, y), *initial_guess)
    initial_guess_data = initial_guess_data - np.min(initial_guess_data)
    initial_guess_data = initial_guess_data / np.max(initial_guess_data)
    initial_guess_data = initial_guess_data * 255
    initial_guess_data = initial_guess_data.astype(np.int8)
    initial_guess_image = Image.fromarray(initial_guess_data.reshape((numy, numx)), mode="L")
    # initial_guess_image.show()

    # use scipy curve fit to fit the gaussian to the data
    print("Fitting gaussian to data... ", end="")
    popt, pcov = opt.curve_fit(gaussian, (x, y), image_data.ravel(), p0=initial_guess)
    print("complete")

    # show the fitted gaussian as an image with data normalized to 0-255
    fitted_data = gaussian((x, y), *popt)
    fitted_data = fitted_data - np.min(fitted_data)
    fitted_data = fitted_data / np.max(fitted_data)
    fitted_data = fitted_data * 255
    fitted_data = fitted_data.astype(np.int8)

    # make the output directory if it doesn't exist
    if not os.path.exists(f"{args.output}/{args.case}"):
        os.makedirs(f"{args.output}/{args.case}")

    # save the composite of the original image and the fitted image
    composite = Image.blend(image, initial_guess_image, 0.5)
    composite.save(f"{args.output}/{args.case}/composite.{image_format.lower()}", image_format)

    # save the fitted data as an image
    fitted_image = Image.fromarray(fitted_data.reshape((numy, numx)), mode="L")
    fitted_image.save(f"{args.output}/{args.case}/fitted.{image_format.lower()}", image_format)

    # save the original image
    image.save(f"{args.output}/{args.case}/original.{image_format.lower()}", image_format)

    # save the fitted parameters
    fitted = GaussianParameters(*popt)
    output = ""
    output += f"A: {fitted.A}\n"
    output += f"mux: {fitted.mux}\n"
    output += f"muy: {fitted.muy}\n"
    output += f"sigmax: {fitted.sigmax * scale * numx} cm\n"
    output += f"sigmay: {fitted.sigmay * scale * numy} cm\n"
    output += f"offset: {fitted.offset}\n"
    with open(f"{args.output}/{args.case}/parameters.txt", "w") as f:
        f.write(output)

    # print the fitted parameters
    print("Fitted parameters:", end="")
    print("\n\t".join(["\n", *output.split("\n")]), end="")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Calculate FWHM of a gaussian spot profile")
    args.add_argument("-c", "--case", required=True, help="path to data case directory")
    args.add_argument("-o", "--output", required=True, help="path to output directory")

    args = args.parse_args()

    main()
