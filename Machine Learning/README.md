# Machine Learning Group 11

## Main Git Commands
 - **git pull** - Fetches most recent files from repository
     - If you haven't worked on the project for a week, most probably someone has and you don't have the most recent version in your PC
     - Should be run every time you start working on the project
     - Usage: `git pull`

  - **git add** - Stages all your changes so they can be committed and pushed to the repository
      - Usage: `git add *` will stage all your changes

  - **git commit** - Commits all your staged changes with a message to be displayed in the repository
      - Usage: `git commit -m "explain what you changed"` will commit all your changes with the message `explain what you changed`
   
  - **git push** - Pushes all your pending commits to the repository
      - Usage: `git push`
   
### The main Git workflow should be:
1. `git pull` to work on the most recent files
2. Do what you have to do, change/add/delete whatever files you deem necessary
3. `git add *` to stage all your changes
4. `git commit -m "changes this and that because of this and that` to explain to the rest of the team what you changed and why
5. `git push` to upload the changes to the repository so the rest of the team can see them
