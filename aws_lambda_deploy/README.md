## Deployment Trilogy
*Example based on Ubuntu 18.04*

### 1. Prerequisite
#### 1.1 Set up AWS CLI
[Reference](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-getting-started-set-up-credentials.html)

#### 1.2 Configure AWS CLI
[Reference](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration)
*You can find the credential-related stuffs at the [IAM Console](https://console.aws.amazon.com/iam/).*

#### 1.3 Set up AWS SAM CLI
[Reference](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install-linux.html)
* Install Homebrew
```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"
test -d ~/.linuxbrew && eval $(~/.linuxbrew/bin/brew shellenv)
test -d /home/linuxbrew/.linuxbrew && eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
test -r ~/.bash_profile && echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.bash_profile
echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.profile
```
* Install AWS SAM CLI
```bash
(sudo apt-get install build-essential
(brew install gcc
(brew tap aws/tap
brew install aws-sam-cli
```
*p.s. Squetially execute the commands with `(` only when you fail when directly execute the last command.*