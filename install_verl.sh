#!/usr/bin/env bash
# install_verl.sh — clone and/or install the `verl` version required by a recipe.
#
# Every recipe directory in this repo ships a REQUIRED_VERL.txt file. This script
# reads that file and either runs the recorded `pip install ...` line (default) or
# performs a full `git clone` + checkout + `pip install -e .` of the upstream verl
# repository at the pinned commit.
#
# Usage:
#   ./install_verl.sh --recipe NAME [options]
#   ./install_verl.sh --file  PATH/TO/REQUIRED_VERL.txt [options]
#   ./install_verl.sh --list
#   ./install_verl.sh --help
#
# Options:
#   --recipe NAME      Recipe directory (e.g. dapo, gkd/megatron, specRL/histoSpec).
#   --file   PATH      Explicit path to a REQUIRED_VERL.txt file.
#   --method pip|git   Install method (default: pip).
#                      * pip: run the recipe's recorded PIP_INSTALL line.
#                      * git: clone UPSTREAM, checkout the pinned commit, then
#                             `pip install -e .` from the checkout.
#   --option NAME      Which variant to pick when the recipe exposes several:
#                        MODE=mixed               -> A | B          (default: B / rolling)
#                        MODE=rolling_with_baseline -> rolling | baseline (default: rolling)
#                        MODE=dapo reproduction    -> rolling | reproduction (default: rolling)
#                      Ignored for other MODEs.
#   --dest DIR         Directory to clone verl into when --method git is used
#                      (default: ./verl).
#   --show             Print the resolved plan and exit without running anything.
#   --yes              Skip the confirmation prompt before executing commands.
#   --list             List every recipe in this repo with its pinned verl spec.
#   --help             Show this help and exit.
#
# Notes:
#   - This script does NOT `source` or `eval` the REQUIRED_VERL.txt file. Values are
#     extracted with `awk` and the commands are printed before being run; with
#     `--show` nothing is executed.
#   - Run from the recipe-repo root (the directory containing this script) so that
#     --recipe NAME resolves to ./NAME/REQUIRED_VERL.txt.
#   - Requires: bash, git, awk, pip (or pip3). `pipx run` or a pre-activated
#     virtualenv is fine; the pip executable is detected automatically.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

print_help() {
    # Print the banner comment (lines 2..41), stripping the leading `# ` / `#`.
    sed -n '2,41p' "${BASH_SOURCE[0]}" | sed -E 's/^#( |$)//'
}

log()  { printf '[install_verl] %s\n' "$*" >&2; }
die()  { printf '[install_verl] error: %s\n' "$*" >&2; exit 1; }

# get_field FILE KEY -> prints the value of the first `KEY=...` line.
# - ignores lines that start with `#`
# - trims leading/trailing whitespace in the value
# - prints nothing (exit 0) if the key is absent
get_field() {
    local file="$1" key="$2"
    awk -v k="$key" '
        /^[[:space:]]*#/ { next }
        {
            line = $0
            sub(/^[[:space:]]+/, "", line)
            n = index(line, "=")
            if (n == 0) next
            name = substr(line, 1, n-1)
            val  = substr(line, n+1)
            sub(/[[:space:]]+$/, "", val)
            if (name == k) { print val; exit }
        }
    ' "$file"
}

detect_pip() {
    if [[ -n "${PIP:-}" ]]; then
        printf '%s\n' "$PIP"
    elif command -v pip >/dev/null 2>&1; then
        printf 'pip\n'
    elif command -v pip3 >/dev/null 2>&1; then
        printf 'pip3\n'
    else
        die "neither pip nor pip3 is on PATH; set PIP=... or activate your env"
    fi
}

# Rewrite `pip install ...` to use the detected pip binary.
rewrite_pip_cmd() {
    local cmd="$1" pip_bin="$2"
    if [[ "$cmd" == pip\ install* ]]; then
        printf '%s %s\n' "$pip_bin" "${cmd#pip }"
    elif [[ "$cmd" == pip3\ install* ]]; then
        printf '%s %s\n' "$pip_bin" "${cmd#pip3 }"
    else
        printf '%s\n' "$cmd"
    fi
}

list_recipes() {
    # shellcheck disable=SC2012  # ls is fine here; we only need a sorted list
    # Find every REQUIRED_VERL.txt and print "recipe_name <tab> pip install line".
    find "$SCRIPT_DIR" -type f -name REQUIRED_VERL.txt \
        -not -path "$SCRIPT_DIR/.git/*" \
        | sort \
        | while read -r f; do
            local rel name mode pip_line
            rel="${f#"$SCRIPT_DIR"/}"
            name="${rel%/REQUIRED_VERL.txt}"
            mode="$(get_field "$f" MODE)"
            pip_line="$(get_field "$f" PIP_INSTALL)"
            if [[ -z "$pip_line" ]]; then
                pip_line="$(get_field "$f" ROLLING_PIP_INSTALL)"
            fi
            if [[ -z "$pip_line" ]]; then
                pip_line="$(get_field "$f" OPTION_B_PIP_INSTALL)"
            fi
            printf '%-32s  MODE=%-24s  %s\n' "$name" "${mode:-?}" "${pip_line:-?}"
        done
}

# resolve_spec FILE OPTION
# Populates these globals (declared in main scope):
#   R_MODE R_UPSTREAM R_COMMIT R_TAG R_PIP_INSTALL R_GIT_SETUP R_LABEL
resolve_spec() {
    local file="$1" option="$2"
    R_MODE="$(get_field "$file" MODE)"
    R_UPSTREAM="$(get_field "$file" UPSTREAM)"
    R_COMMIT=""
    R_TAG=""
    R_PIP_INSTALL=""
    R_GIT_SETUP=""
    R_LABEL=""

    case "$R_MODE" in
        rolling)
            R_LABEL="rolling@$(get_field "$file" VERL_COMMIT)"
            R_COMMIT="$(get_field "$file" VERL_COMMIT)"
            R_PIP_INSTALL="$(get_field "$file" PIP_INSTALL)"
            R_GIT_SETUP="$(get_field "$file" GIT_SETUP)"
            ;;
        pinned_tag)
            R_TAG="$(get_field "$file" TAG)"
            R_COMMIT="$(get_field "$file" COMMIT)"
            R_LABEL="pinned_tag@${R_TAG:-${R_COMMIT}}"
            R_PIP_INSTALL="$(get_field "$file" PIP_INSTALL)"
            R_GIT_SETUP="$(get_field "$file" GIT_SETUP)"
            ;;
        pinned_commit)
            R_COMMIT="$(get_field "$file" COMMIT)"
            R_LABEL="pinned_commit@${R_COMMIT}"
            R_PIP_INSTALL="$(get_field "$file" PIP_INSTALL)"
            R_GIT_SETUP="$(get_field "$file" GIT_SETUP)"
            ;;
        reproduction_commit)
            # DAPO-style file: a fixed reproduction SHA + a rolling main pin.
            local want="${option:-rolling}"
            case "$want" in
                reproduction|repro)
                    R_COMMIT="$(get_field "$file" REPRODUCTION_COMMIT)"
                    R_LABEL="reproduction@${R_COMMIT}"
                    R_PIP_INSTALL="$(get_field "$file" PIP_INSTALL)"
                    ;;
                rolling|"")
                    R_COMMIT="$(get_field "$file" ROLLING_VERL_COMMIT)"
                    [[ -z "$R_COMMIT" ]] && R_COMMIT="$(get_field "$file" VERL_COMMIT)"
                    R_LABEL="rolling@${R_COMMIT}"
                    R_PIP_INSTALL="$(get_field "$file" ROLLING_PIP_INSTALL)"
                    [[ -z "$R_PIP_INSTALL" ]] && R_PIP_INSTALL="$(get_field "$file" PIP_INSTALL)"
                    ;;
                *) die "MODE=reproduction_commit expects --option rolling|reproduction (got '$want')" ;;
            esac
            R_GIT_SETUP="$(get_field "$file" GIT_SETUP)"
            ;;
        mixed)
            local want="${option:-B}"
            case "$want" in
                A|a)
                    R_TAG="$(get_field "$file" OPTION_A_TAG)"
                    R_COMMIT="$(get_field "$file" OPTION_A_COMMIT)"
                    R_PIP_INSTALL="$(get_field "$file" OPTION_A_PIP)"
                    R_LABEL="optionA@${R_TAG:-$R_COMMIT}"
                    ;;
                B|b|rolling|"")
                    R_COMMIT="$(get_field "$file" OPTION_B_VERL_COMMIT)"
                    R_PIP_INSTALL="$(get_field "$file" OPTION_B_PIP_INSTALL)"
                    [[ -z "$R_PIP_INSTALL" ]] && R_PIP_INSTALL="$(get_field "$file" OPTION_B_PIP)"
                    R_LABEL="optionB/rolling@${R_COMMIT}"
                    ;;
                *) die "MODE=mixed expects --option A|B (got '$want')" ;;
            esac
            R_GIT_SETUP="$(get_field "$file" GIT_SETUP)"
            ;;
        rolling_with_baseline)
            local want="${option:-rolling}"
            case "$want" in
                rolling|"")
                    R_COMMIT="$(get_field "$file" ROLLING_VERL_COMMIT)"
                    R_PIP_INSTALL="$(get_field "$file" ROLLING_PIP_INSTALL)"
                    R_GIT_SETUP="$(get_field "$file" ROLLING_GIT_SETUP)"
                    R_LABEL="rolling@${R_COMMIT}"
                    ;;
                baseline|reference)
                    R_TAG="$(get_field "$file" REFERENCE_TAG)"
                    R_COMMIT="$(get_field "$file" REFERENCE_COMMIT)"
                    # No direct PIP line recorded; synthesize one from upstream+commit.
                    if [[ -n "$R_COMMIT" && -n "$R_UPSTREAM" ]]; then
                        R_PIP_INSTALL="pip install verl@git+${R_UPSTREAM}@${R_COMMIT}"
                    fi
                    R_LABEL="baseline@${R_TAG:-$R_COMMIT}"
                    ;;
                *) die "MODE=rolling_with_baseline expects --option rolling|baseline (got '$want')" ;;
            esac
            ;;
        "")
            die "$(printf 'no MODE= field in %s' "$file")"
            ;;
        *)
            # Unknown / future mode. Fall back to whatever PIP_INSTALL is present.
            R_PIP_INSTALL="$(get_field "$file" PIP_INSTALL)"
            R_COMMIT="$(get_field "$file" COMMIT)"
            [[ -z "$R_COMMIT" ]] && R_COMMIT="$(get_field "$file" VERL_COMMIT)"
            R_GIT_SETUP="$(get_field "$file" GIT_SETUP)"
            R_LABEL="${R_MODE}@${R_COMMIT:-unknown}"
            log "unknown MODE=$R_MODE; best-effort using PIP_INSTALL=${R_PIP_INSTALL:-<missing>}"
            ;;
    esac

    [[ -n "$R_UPSTREAM" ]] || R_UPSTREAM="https://github.com/verl-project/verl.git"
}

confirm() {
    local prompt="$1"
    if [[ "${ASSUME_YES:-0}" == "1" ]]; then return 0; fi
    read -r -p "$prompt [y/N] " ans
    [[ "$ans" == "y" || "$ans" == "Y" ]]
}

run_cmd() {
    local cmd="$1"
    log "+ $cmd"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then return 0; fi
    bash -c "$cmd"
}

install_via_pip() {
    local pip_bin rewritten
    if [[ -z "$R_PIP_INSTALL" ]]; then
        die "no PIP_INSTALL line is available for MODE=$R_MODE in this recipe"
    fi
    pip_bin="$(detect_pip)"
    rewritten="$(rewrite_pip_cmd "$R_PIP_INSTALL" "$pip_bin")"
    run_cmd "$rewritten"
}

install_via_git() {
    local dest="$1" pip_bin
    pip_bin="$(detect_pip)"

    # If the recipe recorded a self-contained GIT_SETUP line, run that verbatim
    # (it already does clone + checkout + submodule init + cd, which is the
    # canonical sequence the README expects).
    if [[ -n "$R_GIT_SETUP" ]]; then
        # Only run if the caller didn't override dest away from the default.
        if [[ "$dest" != "verl" && "$dest" != "./verl" ]]; then
            log "note: ignoring recorded GIT_SETUP because --dest=$dest differs from its default"
        else
            run_cmd "$R_GIT_SETUP"
            run_cmd "cd verl && $pip_bin install -e ."
            return
        fi
    fi

    [[ -n "$R_COMMIT" ]] || die "MODE=$R_MODE has no commit to check out; use --method pip instead"

    if [[ -d "$dest/.git" ]]; then
        run_cmd "cd '$dest' && git fetch --tags --depth=1 origin '$R_COMMIT' || git fetch --tags origin"
        run_cmd "cd '$dest' && git checkout '$R_COMMIT'"
    else
        run_cmd "git clone '$R_UPSTREAM' '$dest'"
        run_cmd "cd '$dest' && git checkout '$R_COMMIT'"
    fi
    # Bring the recipe submodule to the matching revision so relative imports work.
    run_cmd "cd '$dest' && git submodule update --init --recursive recipe || true"
    run_cmd "cd '$dest' && $pip_bin install -e ."
}

main() {
    local recipe="" file="" method="pip" option="" dest="verl" show=0
    while (($#)); do
        case "$1" in
            --recipe) recipe="${2:?}"; shift 2 ;;
            --file)   file="${2:?}";   shift 2 ;;
            --method) method="${2:?}"; shift 2 ;;
            --option) option="${2:?}"; shift 2 ;;
            --dest)   dest="${2:?}";   shift 2 ;;
            --show)   show=1; shift ;;
            --yes|-y) ASSUME_YES=1; shift ;;
            --list)   list_recipes; exit 0 ;;
            -h|--help) print_help; exit 0 ;;
            *) die "unknown argument: $1 (try --help)" ;;
        esac
    done

    if [[ -z "$recipe" && -z "$file" ]]; then
        print_help
        exit 2
    fi

    if [[ -z "$file" ]]; then
        file="$SCRIPT_DIR/$recipe/REQUIRED_VERL.txt"
    fi
    [[ -f "$file" ]] || die "REQUIRED_VERL.txt not found at: $file"

    # Declared here so resolve_spec can set them in this scope.
    local R_MODE R_UPSTREAM R_COMMIT R_TAG R_PIP_INSTALL R_GIT_SETUP R_LABEL
    resolve_spec "$file" "$option"

    cat >&2 <<EOF
[install_verl] file:     $file
[install_verl] MODE:     $R_MODE
[install_verl] upstream: $R_UPSTREAM
[install_verl] resolved: $R_LABEL
[install_verl] method:   $method
[install_verl] pip line: ${R_PIP_INSTALL:-<none>}
EOF

    if [[ "$show" == "1" ]]; then
        DRY_RUN=1
        case "$method" in
            pip) install_via_pip ;;
            git) install_via_git "$dest" ;;
            *)   die "--method must be pip or git (got '$method')" ;;
        esac
        return 0
    fi

    case "$method" in
        pip)
            if ! confirm "Proceed with pip install?"; then die "aborted"; fi
            install_via_pip
            ;;
        git)
            if ! confirm "Proceed with git clone into '$dest' + pip install -e .?"; then die "aborted"; fi
            install_via_git "$dest"
            ;;
        *) die "--method must be pip or git (got '$method')" ;;
    esac

    log "done."
}

main "$@"
