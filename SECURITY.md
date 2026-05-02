# Security Policy

## Supported Versions

Security fixes are prioritized for the active development branch and the latest stable release.

| Version or branch | Security support |
| --- | --- |
| `main` | Supported |
| Latest stable release | Supported |
| Older releases | Best effort |

If you are unsure whether a version is supported, report the issue anyway and include the package version, commit SHA, installation method, and operating system.

## Reporting a Vulnerability

Please do not open a public GitHub issue for suspected vulnerabilities.

Use one of these private channels instead:

1. GitHub private vulnerability reporting: https://github.com/TensorAeroSpace/TensorAeroSpace/security/advisories/new
2. Email fallback: support@tensoraerospace.org with the subject prefix `[SECURITY]`

Include as much detail as you can safely share:

- Affected package version, commit SHA, or branch.
- Environment details such as OS, Python version, CUDA/GPU setup, and installation method.
- A minimal reproduction or proof of concept.
- Expected impact and affected components.
- Whether the issue is already public or under coordinated disclosure elsewhere.

## Response Expectations

Maintainers will try to acknowledge reports within 2 business days. After triage, we will aim to provide an initial assessment within 14 days, including whether the report is accepted, needs more information, or is outside the project scope.

If a fix is needed, we will coordinate disclosure timing with the reporter when possible. Please give maintainers reasonable time to investigate and release a patch before publishing details.

## Scope

Security reports are in scope when they affect TensorAeroSpace code, packaging, documentation that leads users to unsafe behavior, CI/CD configuration, or official project artifacts.

Examples of useful reports include:

- Arbitrary code execution or unsafe model/checkpoint loading paths.
- Credential, token, or secret exposure.
- Dependency, packaging, or release-process issues that affect users.
- Unsafe deserialization, path traversal, command injection, or file overwrite behavior.
- Documentation that instructs users to perform an unsafe security-sensitive action.

Out-of-scope reports usually include:

- Vulnerabilities only present in unsupported third-party services or local user systems.
- Social engineering against maintainers or contributors.
- Denial-of-service claims without a practical impact on the project or its users.
- Reports generated only by automated scanners without explanation or reproduction.

## Public Disclosure

Do not disclose vulnerability details publicly until maintainers have completed triage and, when needed, released a fix or mitigation. If a vulnerability is already public, please still report it privately so maintainers can track the project impact and remediation.

## Security Baselines

The repository uses baseline-based security gates for known static-analysis and dependency findings. Passing these gates does not mean the project is free of vulnerabilities. New reports are still welcome, especially when they include practical impact or a reproducible exploit path.
