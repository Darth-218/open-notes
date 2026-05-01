# Shell Scripting

## Basic Commands

- `ls` - list files
- `cd` - change directory
- `pwd` - print working directory
- `cat` - display file contents
- `grep` - search patterns

## Variables

```bash
NAME="Alice"
echo "Hello, $NAME"
```

## Conditionals

```bash
if [ -f "$FILE" ]; then
    echo "File exists"
else
    echo "File not found"
fi
```

## Loops

```bash
for file in *.txt; do
    echo "Processing $file"
done
```

## Functions

```bash
greet() {
    echo "Hello, $1!"
}

greet "World"
```