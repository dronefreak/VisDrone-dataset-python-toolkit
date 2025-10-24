# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

We take the security of VisDrone Toolkit seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please Do Not

- **Do not** open a public GitHub issue for security vulnerabilities
- **Do not** discuss the vulnerability publicly until it has been addressed

### Please Do

**Report security vulnerabilities via email to: <your.email@example.com>**

Include the following information:

- Type of vulnerability
- Full paths of source file(s) related to the manifestation of the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability, including how an attacker might exploit it

### What to Expect

After you submit a report:

1. **Acknowledgment**: We will acknowledge receipt within 48 hours
2. **Assessment**: We will assess the vulnerability and determine its impact
3. **Resolution**: We will work on a fix and coordinate with you on disclosure
4. **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)

### Disclosure Policy

- Security vulnerabilities will be disclosed once a fix is available
- We aim to patch critical vulnerabilities within 30 days
- We will coordinate with you on the disclosure timeline

## Security Best Practices

### When Using This Toolkit

1. **Keep Dependencies Updated**

   ```bash
   pip install --upgrade visdrone-toolkit
   pip list --outdated
   ```

2. **Validate Input Data**

   - Always validate datasets before training
   - Check for malicious files in uploaded datasets
   - Use trusted data sources

3. **Secure Model Checkpoints**

   - Only load checkpoints from trusted sources
   - Verify checkpoint integrity before loading
   - Be cautious with user-uploaded model files

4. **Environment Security**

   - Use virtual environments
   - Keep PyTorch and dependencies updated
   - Review security advisories for dependencies

5. **API Keys and Credentials**
   - Never commit API keys or credentials
   - Use environment variables for sensitive data
   - Add `.env` files to `.gitignore`

### Known Security Considerations

#### Model Loading

- Loading untrusted model checkpoints can execute arbitrary code
- Always verify the source of model files
- Consider using `torch.load(..., weights_only=True)` when possible

#### Data Processing

- Image processing libraries may have vulnerabilities
- Keep PIL, OpenCV, and other image libraries updated
- Validate image files before processing

#### Web Services

- If deploying as a web service, implement proper authentication
- Use HTTPS for all communications
- Implement rate limiting to prevent abuse

## Security Updates

Security updates will be released as patch versions (e.g., 2.0.1, 2.0.2). We recommend:

- Subscribe to GitHub releases
- Watch the repository for security advisories
- Check CHANGELOG.md regularly

## Vulnerability Disclosure Timeline

1. **Day 0**: Vulnerability reported to maintainers
2. **Day 1-2**: Acknowledgment sent to reporter
3. **Day 3-7**: Vulnerability assessed and validated
4. **Day 7-30**: Fix developed and tested
5. **Day 30**: Security advisory published, patch released

## Past Security Advisories

None yet. This is version 2.0.

## Security Contact

- **Email**: <your.email@example.com>
- **PGP Key**: [Coming soon]

## Bug Bounty Program

We currently do not have a bug bounty program, but we appreciate and acknowledge security researchers who responsibly disclose vulnerabilities.

## Additional Resources

- [OWASP Top Ten](https://owasp.org/www-project-top-ten/)
- [PyTorch Security](https://pytorch.org/docs/stable/security.html)
- [Python Security](https://python.org/dev/security/)

## Acknowledgments

We thank the following security researchers for their responsible disclosure:

- [List will be updated as vulnerabilities are reported and fixed]

---

**Thank you for helping keep VisDrone Toolkit and our users safe!**
