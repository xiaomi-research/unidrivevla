import json

# Please fill in your team information here
method = "ReCogDrive"  # <str> -- name of the method
team = "ReCogDrive-XiamiEV"  # <str> -- name of the team, !!!identical to the Google Form!!!
authors = ["Yongkang Li"]  # <list> -- list of str, authors
email = "liyk@hust.edu.cn"  # <str> -- e-mail address
institution = "XiamiEV"  # <str> -- institution or company
country = "China"  # <str> -- country or region


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='path to model output json')
    args = parser.parse_args()

    with open(args.input, 'r') as file:
        output_res = json.load(file)

    submission_content = {
        "method": method,
        "team": team,
        "authors": authors,
        "email": email,
        "institution": institution,
        "country": country,
        "results": output_res
    }

    with open('submission.json', 'w') as file:
        json.dump(submission_content, file, indent=4)

if __name__ == "__main__":
    main()
